"""Logarithmic energy minimization of 150 points on the unit sphere.
Uses a multi-start strategy with Velocity-Verlet, Adam, and L-BFGS-B polishing."""

import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    The gradient is projected to the tangent space of the sphere.
    """
    n = pts.shape[0]
    # Compute pairwise distances squared
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist2, 1.0)
    
    # Compute energy
    # We only need the upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())
    
    # Compute gradient
    # grad_i = sum_j (x_j - x_i) / dist_ij^2
    # Note: 1/dist2 is symmetric
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad = inv_dist2 @ pts - pts * row_sums
    # row_sums[i] = sum_j inv_dist2[i,j]
    row_sums = np.sum(inv_dist2, axis=1)
    grad = inv_dist2 @ pts
    grad -= pts * row_sums[:, np.newaxis]
    
    # Project gradient to tangent space
    # tangent_grad = grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _energy_only(pts):
    """Compute only the energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    """Generate points using a Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron (geodesic dome),
    then project to sphere. Provides a different structure than Fibonacci spiral.
    """
    rng = np.random.RandomState(seed)
    # Start with icosahedron vertices
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Edges of icosahedron
    edges = [(0,11),(0,5),(0,1),(0,7),(0,10),(1,5),(5,11),(11,10),(10,7),
             (7,1),(1,6),(6,10),(2,11),(2,5),(2,9),(2,4),(2,8),(3,5),
             (3,11),(3,4),(3,9),(3,6),(4,9),(5,4),(11,4),(10,7),(6,1),
             (8,4),(9,8),(8,5),(6,9),(7,2),(8,3)]
    
    # Recursively subdivide until we have enough points
    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])
    
    # If we have more than n, randomly select n
    if len(verts) > n:
        idx = rng.choice(len(verts), size=n, replace=False)
        verts = verts[idx]
    
    return verts


def _random_points(n, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _adam_optimize(pts0, max_iter=3000, lr_start=0.01, lr_end=0.001,
                   beta1=0.9, beta2=0.999, eps=1e-8, deadline=None, check_interval=20):
    """
    Adam optimization on the sphere with learning rate decay.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    
    # Cosine decay for learning rate
    def lr_schedule(t):
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t / max_iter))

    for t in range(1, max_iter + 1):
        if deadline is not None and (t % check_interval == 0 or t == 1):
            if time.time() > deadline:
                break

        energy, grad = _energy_and_grad(pts)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        lr = lr_schedule(t)
        
        # Step in tangent direction
        pts_new = pts - lr * update
        pts_new = _project_to_sphere(pts_new)

        if not np.all(np.isfinite(pts_new)):
            break

        energy_new = _energy_only(pts_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
            
        pts = pts_new

    return best_pts, best_energy


def _velocity_verlet_sphere(points, deadline, dt=0.005, n_steps=5000, damping=0.98):
    """
    Velocity-Verlet integration on the sphere.
    """
    pts = _project_to_sphere(points.copy())
    vel = np.zeros_like(pts)
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Initial force
    _, grad0 = _energy_and_grad(pts)
    force = -grad0
    
    for step in range(n_steps):
        if time.time() > deadline - 0.5:
            break
        
        # Velocity-Verlet half step
        vel_half = vel + 0.5 * dt * force
        # Project velocity to tangent space
        dot_v = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half = vel_half - dot_v * pts
        
        # Position update
        pts_new = pts + dt * vel_half
        pts_new = _project_to_sphere(pts_new)
        
        # New force
        _, grad_new = _energy_and_grad(pts_new)
        force_new = -grad_new
        # Project force to tangent space
        dot_f = np.sum(force_new * pts_new, axis=1, keepdims=True)
        force_new = force_new - dot_f * pts_new
        
        # Velocity update with damping
        vel = (vel_half + 0.5 * dt * force_new) * damping
        dot_v2 = np.sum(vel * pts_new, axis=1, keepdims=True)
        vel = vel - dot_v2 * pts_new
        
        pts = pts_new
        force = force_new
        
        energy = _energy_only(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Adaptive damping or step size if energy increases
        if step > 100 and energy > best_energy * 1.05:
            dt *= 0.9
            damping *= 0.99

    return best_pts, best_energy


def _lbfgs_polish(pts, deadline, maxiter=500):
    """
    L-BFGS-B polish using flat Cartesian coordinates with projection.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 30,
                                'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        if e_check < _energy_only(pts):
            return p_res
    except Exception:
        pass
    return pts


def _lbfgs_tight_polish(pts, deadline, maxiter=1000):
    """
    Ultra-tight L-BFGS-B polish.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 50,
                                'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 60})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        if e_check < _energy_only(pts):
            return p_res
    except Exception:
        pass
    return pts


def _local_perturb_restart(pts, rng, perturb_scale=0.05, n_perturb=10):
    """
    Create perturbed versions of pts by randomly moving n_perturb points.
    Returns list of perturbed point arrays.
    """
    results = []
    for _ in range(5):
        p = pts.copy()
        idx = rng.choice(N, size=n_perturb, replace=False)
        noise = rng.randn(n_perturb, 3) * perturb_scale
        p[idx] += noise
        p = _project_to_sphere(p)
        results.append(p)
    return results


def _simulated_annealing_tempered(pts_list, t_init, t_min, n_sweeps, deadline,
                                   n_replicas=4, swap_interval=20,
                                   move_scale=0.03, move_scale_end=0.005):
    """
    Parallel tempering (replica exchange) with simulated annealing.
    
    pts_list: list of point configurations (one per replica)
    t_init, t_min: initial and final temperatures
    n_sweeps: number of sweeps (each sweep = one random point move per replica)
    n_replicas: number of replicas (should match len(pts_list))
    swap_interval: how often to attempt replica swaps
    move_scale: initial random displacement scale
    move_scale_end: final random displacement scale
    """
    n_replicas = len(pts_list)
    energies = [_energy_only(p) for p in pts_list]
    best_energy = min(energies)
    best_pts = pts_list[np.argmin(energies)].copy()
    
    # Temperature ladder: geometric spacing
    temp_ratio = (t_min / t_init) ** (1.0 / n_replicas)
    temps = [t_init * (temp_ratio ** i) for i in range(n_replicas)]
    # Sort so replica 0 is hottest, replica n-1 is coldest
    # Actually we want replica 0 = cold, replica n-1 = hot for standard convention
    # Let's use: replica 0 = coldest, replica n-1 = hottest
    temps = [t_init * (temp_ratio ** (n_replicas - 1 - i)) for i in range(n_replicas)]
    
    rng = np.random.RandomState(12345)
    
    for sweep in range(n_sweeps):
        if time.time() > deadline - 1.0:
            break
        
        # Scale move size with temperature
        frac = sweep / max(n_sweeps - 1, 1)
        scale = move_scale_end + (move_scale - move_scale_end) * (1 - frac)
        
        # Each replica makes one random move
        for r in range(n_replicas):
            p = pts_list[r]
            e_old = energies[r]
            
            # Pick a random point and displace it
            idx = rng.randint(N)
            displacement = rng.randn(3) * scale
            new_point = p[idx] + displacement
            new_point = new_point / np.linalg.norm(new_point)
            
            p_new = p.copy()
            p_new[idx] = new_point
            
            e_new = _energy_only(p_new)
            
            # Metropolis criterion
            delta_e = e_new - e_old
            if delta_e < 0 or rng.random() < math.exp(-delta_e / temps[r]):
                pts_list[r] = p_new
                energies[r] = e_new
                
                if e_new < best_energy:
                    best_energy = e_new
                    best_pts = p_new.copy()
        
        # Attempt replica exchange (swap)
        if sweep % swap_interval == 0 and n_replicas > 1:
            # Try swapping adjacent replicas
            r1 = rng.randint(n_replicas - 1)
            r2 = r1 + 1
            
            e1 = energies[r1]
            e2 = energies[r2]
            t1 = temps[r1]
            t2 = temps[r2]
            
            # Compute acceptance probability for swap
            # P_swap = min(1, exp(-(e1/t2 + e2/t1 - e1/t1 - e2/t2)))
            exponent = -(e1 / t2 + e2 / t1 - e1 / t1 - e2 / t2)
            if exponent > 0 or rng.random() < math.exp(min(exponent, 0)):
                # Swap
                pts_list[r1], pts_list[r2] = pts_list[r2], pts_list[r1]
                energies[r1], energies[r2] = energies[r2], energies[r1]
                
                # Update best if needed
                for r in range(n_replicas):
                    if energies[r] < best_energy:
                        best_energy = energies[r]
                        best_pts = pts_list[r].copy()
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    # Define multiple initializations
    # 1. Fibonacci spiral
    init1 = _fibonacci_spiral(N)
    # 2. Fibonacci spiral with different twist
    init2 = _fibonacci_spiral(N, twist=1.5)
    # 3. Random points
    init3 = _random_points(N, seed=123)
    # 4. Random points different seed
    init4 = _random_points(N, seed=456)
    # 5. Perturbed Fibonacci
    rng = np.random.RandomState(999)
    init5 = _project_to_sphere(_fibonacci_spiral(N) + rng.randn(N, 3) * 0.1)
    # 6. Geodesic dome
    init6 = _geodesic_dome(N, seed=7)

    inits = [init1, init2, init3, init4, init5, init6]
    
    best_pts_global = None
    best_energy_global = float('inf')

    # Time per restart
    # We have ~140s. Let's do 4 main restarts, each ~30s, and keep the best.
    # Then use remaining time for local perturbation restarts and final polish.
    
    time_per_restart = 30.0
    num_restarts = 4
    
    for i in range(num_restarts):
        current_deadline = start_time + (i + 1) * time_per_restart
        if current_deadline > deadline - 30.0: # Leave 30s for perturbation restarts + final polish
            current_deadline = deadline - 30.0
            
        pts_start = inits[i % len(inits)]
        
        # Phase 1: Velocity Verlet (fast basin finding)
        vv_deadline = current_deadline - 12.0
        if time.time() < vv_deadline:
            pts_vv, e_vv = _velocity_verlet_sphere(pts_start, vv_deadline, dt=0.005, n_steps=2500)
        else:
            pts_vv, e_vv = pts_start, _energy_only(pts_start)
            
        # Phase 2: Adam (refine)
        adam_deadline = current_deadline - 4.0
        if time.time() < adam_deadline:
            pts_adam, e_adam = _adam_optimize(pts_vv, max_iter=1200, lr_start=0.005, lr_end=0.0005, deadline=adam_deadline)
        else:
            pts_adam, e_adam = pts_vv, e_vv
            
        # Phase 3: L-BFGS-B (polish)
        if time.time() < current_deadline - 1.0:
            pts_lbfgs = _lbfgs_polish(pts_adam, current_deadline - 1.0, maxiter=300)
            e_lbfgs = _energy_only(pts_lbfgs)
        else:
            pts_lbfgs, e_lbfgs = pts_adam, e_adam
            
        if e_lbfgs < best_energy_global:
            best_energy_global = e_lbfgs
            best_pts_global = pts_lbfgs.copy()

    # Local Perturbation Restart Phase
    # Try perturbing the best solution to escape local minima
    if best_pts_global is not None and time.time() < deadline - 10.0:
        perturb_deadline = deadline - 10.0
        rng_perturb = np.random.RandomState(42)
        
        # First: try parallel tempering / simulated annealing from best solution
        if time.time() < perturb_deadline - 5.0:
            # Create replicas with varying amounts of noise
            base = best_pts_global
            replicas = []
            for r in range(4):
                noise_scale = 0.02 * (r + 1)
                p_rep = _project_to_sphere(base + rng_perturb.randn(N, 3) * noise_scale)
                replicas.append(p_rep)
            
            t_init = 0.5
            t_min = 0.01
            n_sweeps = 5000
            sa_deadline = perturb_deadline - 3.0
            
            pts_sa, e_sa = _simulated_annealing_tempered(
                replicas, t_init=t_init, t_min=t_min,
                n_sweeps=n_sweeps, deadline=sa_deadline,
                n_replicas=4, swap_interval=20,
                move_scale=0.05, move_scale_end=0.003)
            
            if e_sa < best_energy_global:
                best_energy_global = e_sa
                best_pts_global = pts_sa.copy()
            
            # Polish the SA result
            if time.time() < sa_deadline - 1.0:
                pts_pol = _lbfgs_polish(best_pts_global, sa_deadline - 1.0, maxiter=200)
                e_pol = _energy_only(pts_pol)
                if e_pol < best_energy_global:
                    best_energy_global = e_pol
                    best_pts_global = pts_pol.copy()
        
        # Then: try several perturbations with decreasing scale
        for scale in [0.08, 0.05, 0.03]:
            if time.time() > perturb_deadline - 5.0:
                break
            
            perturbations = _local_perturb_restart(best_pts_global, rng_perturb, 
                                                   perturb_scale=scale, n_perturb=8)
            for pert_pts in perturbations:
                if time.time() > perturb_deadline - 2.0:
                    break
                
                # Quick Adam refinement
                if time.time() < perturb_deadline - 3.0:
                    pts_ref, e_ref = _adam_optimize(pert_pts, max_iter=500, 
                                                     lr_start=0.003, lr_end=0.0003,
                                                     deadline=perturb_deadline - 3.0)
                    # Quick L-BFGS polish
                    if time.time() < perturb_deadline - 1.0:
                        pts_pol = _lbfgs_polish(pts_ref, perturb_deadline - 1.0, maxiter=100)
                        e_pol = _energy_only(pts_pol)
                    else:
                        pts_pol, e_pol = pts_ref, e_ref
                    
                    if e_pol < best_energy_global:
                        best_energy_global = e_pol
                        best_pts_global = pts_pol.copy()

    # Final Polish Phase
    # Use remaining time to polish the global best
    remaining_deadline = deadline - 2.0
    
    if best_pts_global is None:
        best_pts_global = _fibonacci_spiral(N)
        best_energy_global = _energy_only(best_pts_global)
        
    # Tight L-BFGS with more iterations
    if time.time() < remaining_deadline - 3.0:
        best_pts_global = _lbfgs_tight_polish(best_pts_global, remaining_deadline - 3.0, maxiter=1500)
        best_energy_global = _energy_only(best_pts_global)
        
    # One more standard L-BFGS with different settings
    if time.time() < remaining_deadline - 0.5:
        best_pts_global = _lbfgs_polish(best_pts_global, remaining_deadline - 0.5, maxiter=300)
        best_energy_global = _energy_only(best_pts_global)

    best_pts_global = _project_to_sphere(best_pts_global)

    # Validation
    if not np.all(np.isfinite(best_pts_global)):
        return _fibonacci_spiral(N)

    diff = best_pts_global[:, np.newaxis, :] - best_pts_global[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_spiral(N)

    return best_pts_global

# EVOLVE-BLOCK-END
