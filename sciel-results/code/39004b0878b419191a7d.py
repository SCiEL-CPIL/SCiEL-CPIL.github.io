"""Packing 50 equal circles in a unit circle using L-BFGS-B on a soft-min surrogate."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START
def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.125
    s = 2.0 * r_est
    sqrt3 = np.sqrt(3.0)
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * sqrt3 / 2.0
            if np.hypot(x, y) <= 0.98:
                points.append([x, y])
                
    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _get_ring_initialization(n, num_rings=4):
    """Generate ring-based initialization with better spacing."""
    centers = []
    centers.append([0.0, 0.0])
    
    remaining = n - 1
    # Distribute points across rings with increasing radius
    ring_radii = np.linspace(0.25, 0.9, num_rings)
    points_per_ring = [remaining // num_rings] * num_rings
    for i in range(remaining - sum(points_per_ring)):
        points_per_ring[i % num_rings] += 1
    
    for ring_idx, count in enumerate(points_per_ring):
        if count <= 0:
            continue
        radius = ring_radii[ring_idx]
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + ring_idx * 0.3
        for angle in angles:
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    return np.array(centers[:n])

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.88)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _get_fibonacci_initialization(n):
    """Generate Fibonacci-sphere-like initialization projected to disk."""
    centers = []
    centers.append([0.0, 0.0])
    remaining = n - 1
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(1, remaining + 1):
        # Distribute points in rings with Fibonacci angle offset
        ring = i
        count_in_ring = 6  # hexagonal
        idx_in_ring = i - 1
        radius = 0.2 + 0.65 * np.sqrt(idx_in_ring / max(remaining, 1))
        angle = idx_in_ring * 2 * np.pi / golden_ratio
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers[:n])

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _soft_min(values, beta=50.0):
    """Compute the soft minimum of an array of values."""
    v = np.array(values)
    m = np.min(v)
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))

def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n, 2) array of centers
    Returns (objective, gradient) for L-BFGS-B.
    """
    centers = params.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    obj = -_soft_min(vals, beta)
    
    exp_vals = np.exp(-beta * (vals - np.min(vals)))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    # Pair contributions (vectorized)
    pair_weights = weights[:n*(n-1)//2]
    d_vecs = centers[i_idx] - centers[j_idx]
    d_mags = pair_dists
    safe_mags = np.where(d_mags > 1e-10, d_mags, 1.0)
    unit_vecs = d_vecs / safe_mags[:, np.newaxis]
    
    # For each pair k: grad[i] += w_k * unit_vec[k] / 2, grad[j] -= w_k * unit_vec[k] / 2
    # Vectorize using np.add.at
    half_w = pair_weights / 2.0
    np.add.at(grad, i_idx, half_w[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, -half_w[:, np.newaxis] * unit_vecs)
    
    # Boundary contributions
    bound_weights = weights[n*(n-1)//2:]
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad -= (bound_weights / safe_norms)[:, np.newaxis] * centers
    
    return obj, -grad

def _optimize_from_start(centers, beta, maxiter=100, gtol=1e-8):
    """Optimize from a given starting point using L-BFGS-B."""
    try:
        res = minimize(
            _objective_and_gradient, 
            centers.flatten(), 
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers

def _nelder_mead_polish(centers, maxiter=200, tol=1e-10):
    """Polish using Nelder-Mead on the raw (nonsmooth) objective."""
    def raw_obj(params):
        c = params.reshape(N, 2)
        return -_compute_score(c)
    
    try:
        res = minimize(
            raw_obj,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers

def _rigid_body_dynamics(centers, start_time, dt=0.002, max_steps=5000):
    """
    Rigid-body dynamics with velocity-Verlet integration and adaptive damping.
    Uses strong short-range repulsive forces and harmonic confinement to drive
    the system toward a jammed packing state. Momentum helps escape local minima.
    """
    pos = centers.copy().astype(np.float64)
    vel = np.zeros_like(pos)
    
    # Force parameters
    k_repel = 1000.0   # repulsion strength
    r_target = _compute_score(pos)  # target separation
    k_confinement = 50.0  # harmonic potential strength
    r_confine = 1.0 - r_target - 0.01  # confinement radius
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def compute_forces(positions):
        """Compute all forces on each particle."""
        forces = np.zeros_like(positions)
        
        # Pairwise repulsive forces (short-range, strong)
        d_vecs = positions[i_idx] - positions[j_idx]
        dists = np.linalg.norm(d_vecs, axis=1)
        safe_dists = np.maximum(dists, 1e-10)
        
        # Only apply repulsion when closer than 2*r_target
        overlap = np.maximum(0, 2 * r_target - dists)
        force_mags = k_repel * overlap / safe_dists
        unit_vecs = d_vecs / safe_dists[:, np.newaxis]
        
        # Apply forces to pairs
        np.add.at(forces, i_idx, force_mags[:, np.newaxis] * unit_vecs)
        np.add.at(forces, j_idx, -force_mags[:, np.newaxis] * unit_vecs)
        
        # Confinement: harmonic potential pulling toward center
        norms = np.linalg.norm(positions, axis=1)
        safe_norms = np.maximum(norms, 1e-10)
        # Only confine if outside r_confine
        confine_overlap = np.maximum(0, norms - r_confine)
        confine_force_mags = k_confinement * confine_overlap / safe_norms
        forces -= confine_force_mags[:, np.newaxis] * positions / safe_norms[:, np.newaxis]
        
        return forces
    
    # Velocity-Verlet with adaptive damping
    damping = 0.95  # initial damping coefficient
    best_score = _compute_score(pos)
    best_pos = pos.copy()
    
    for step in range(max_steps):
        if time.time() - start_time > _DEADLINE - 8:
            break
        
        # Compute forces at current position
        f = compute_forces(pos)
        
        # Velocity-Verlet integration
        # Half-step velocity
        vel_half = vel + 0.5 * dt * f
        
        # Update position
        pos_new = pos + dt * vel_half
        
        # Compute forces at new position
        f_new = compute_forces(pos_new)
        
        # Full-step velocity with damping
        vel_new = (vel_half + 0.5 * dt * f_new) * damping
        
        # Keep positions bounded
        norms_new = np.linalg.norm(pos_new, axis=1)
        max_r = 1.0 - 0.005
        mask = norms_new > max_r
        if np.any(mask):
            pos_new[mask] *= max_r / norms_new[mask]
            vel_new[mask] *= 0.5  # damp velocity at boundary
        
        pos = pos_new
        vel = vel_new
        
        # Adaptively reduce damping as system settles
        damping = max(0.999, damping * 0.9995)
        
        # Track best
        current_score = _compute_score(pos)
        if current_score > best_score:
            best_score = current_score
            best_pos = pos.copy()
            # Update target radius for better repulsion
            r_target = best_score
    
    return best_pos, best_score

def _polish(centers, start_time):
    """Progressively polish the solution with increasing beta, rigid-body dynamics, and NM."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    # Phase 1: Soft-min gradient optimization with increasing beta
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1500.0]
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=60, gtol=1e-9)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    # Phase 2: Rigid-body dynamics to escape local minima
    if time.time() - start_time < _DEADLINE - 15:
        rb_result, rb_score = _rigid_body_dynamics(current, start_time, dt=0.001, max_steps=3000)
        if np.all(np.isfinite(rb_result)):
            if rb_score > best_score:
                best_score = rb_score
                current = rb_result
        
        # Re-optimize with soft-min after dynamics perturbation
        if time.time() - start_time < _DEADLINE - 10:
            polished = _optimize_from_start(current, beta=2000.0, maxiter=80, gtol=1e-10)
            if np.all(np.isfinite(polished)):
                r = _compute_score(polished)
                if r > best_score:
                    best_score = r
                    current = polished
    
    # Phase 3: High-precision final polish
    if time.time() - start_time < _DEADLINE - 5:
        # Very high beta soft-min for precise convergence
        polished = _optimize_from_start(current, beta=10000.0, maxiter=50, gtol=1e-12)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    # Phase 4: Nelder-Mead on the nonsmooth objective for final adjustments
    if time.time() - start_time < _DEADLINE - 3:
        nm_result = _nelder_mead_polish(current, maxiter=200, tol=1e-12)
        if np.all(np.isfinite(nm_result)):
            r = _compute_score(nm_result)
            if r > best_score:
                best_score = r
                current = nm_result
    
    return current, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Multiple diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=999),
        _get_fibonacci_initialization(N),
    ]
    
    # First pass: optimize each initialization with moderate beta
    for init in initializations:
        if time.time() - start_time > _DEADLINE - 35:
            break
        
        # Ensure all points are inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.98 / norms[mask]
        
        optimized = _optimize_from_start(init, beta=50.0, maxiter=100, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
    
    # Second pass: polish the best solution
    if best_centers is not None and time.time() - start_time < _DEADLINE - 25:
        best_centers, best_score = _polish(best_centers, start_time)
    
    # Third pass: try to improve with perturbations of the best
    if best_centers is not None and time.time() - start_time < _DEADLINE - 20:
        rng = np.random.default_rng(7)
        for attempt in range(10):
            if time.time() - start_time > _DEADLINE - 20:
                break
            
            noise_scale = 0.003 + 0.008 * attempt
            noisy = best_centers + rng.normal(0, noise_scale, best_centers.shape)
            norms = np.linalg.norm(noisy, axis=1)
            mask = norms > 0.98
            if np.any(mask):
                noisy[mask] *= 0.98 / norms[mask]
            
            # Use rigid-body dynamics from perturbed start
            rb_result, rb_score = _rigid_body_dynamics(noisy, start_time, dt=0.001, max_steps=1500)
            if np.all(np.isfinite(rb_result)):
                # Polish the dynamics result
                polished = _optimize_from_start(rb_result, beta=500.0, maxiter=50, gtol=1e-9)
                if np.all(np.isfinite(polished)):
                    r = _compute_score(polished)
                    if r > best_score:
                        best_score = r
                        best_centers = polished.copy()
        
        # Final high-precision polish
        if time.time() - start_time < _DEADLINE - 5:
            best_centers, best_score = _polish(best_centers, start_time)
    
    if best_centers is None:
        return initializations[0]
        
    return best_centers
# EVOLVE-BLOCK-END
