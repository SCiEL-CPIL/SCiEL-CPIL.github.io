"""
Circle packing in a unit disk using rigid-body dynamics with symplectic integration,
collision detection, impulse-based resolution, and L-BFGS-B polish.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE_SECONDS = 138.0  # safety margin before 150s kill


def _score(points):
    """Compute the score: min of half min pairwise distance and min margin to boundary."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists_to_center = np.sqrt(np.sum(points**2, axis=1))
    if np.any(dists_to_center > 1.0 + 1e-12):
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_margin = 1.0 - np.max(dists_to_center)
    r = min(min_pair_dist / 2.0, min_margin)
    if r < 0:
        return 0.0
    return r


def _soft_min(values, beta):
    """Soft minimum: -log(sum(exp(-beta*v)))/beta approximates min(v) for large beta."""
    v = np.asarray(values)
    if v.size == 0:
        return 0.0
    shifted = v - np.min(v)
    return -np.log(np.sum(np.exp(-beta * shifted))) / beta + np.min(v)


def _objective_and_gradient(points_flat, beta=100.0):
    """
    Objective: negative soft-min of (pairwise distances / 2, boundary margins).
    Returns (value, gradient).
    """
    points = points_flat.reshape(N, 2)
    
    # Boundary margins
    r_i = np.sqrt(np.sum(points**2, axis=1))
    margins_boundary = 1.0 - r_i
    
    # Pairwise distances / 2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    half_dists = dists / 2.0
    # Flatten upper triangle
    iu = np.triu_indices(N, k=1)
    half_dists_flat = half_dists[iu]
    
    # All values for soft-min
    all_vals = np.concatenate([half_dists_flat, margins_boundary])
    
    # Soft min
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    soft_min_val = min_val - np.log(sum_exp) / beta
    
    # Gradient computation
    # d(soft_min)/d(all_vals[k]) = -exp_vals[k] / sum_exp
    weights = -exp_vals / sum_exp  # shape: (N*(N-1)/2 + N,)
    
    grad = np.zeros_like(points_flat)
    
    # Boundary part: last N entries
    for i in range(N):
        w = weights[N*(N-1)//2 + i]
        # d(1 - r_i)/d(points[i]) = -points[i] / r_i
        if r_i[i] > 1e-12:
            grad[2*i] = w * (-points[i, 0] / r_i[i])
            grad[2*i+1] = w * (-points[i, 1] / r_i[i])
    
    # Pairwise part: first N*(N-1)/2 entries
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            w = weights[idx]
            # d(|p_i - p_j|/2)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|)
            d = points[i] - points[j]
            dist = np.sqrt(np.sum(d**2))
            if dist > 1e-12:
                grad[2*i] += w * d[0] / dist
                grad[2*i+1] += w * d[1] / dist
                grad[2*j] += w * (-d[0]) / dist
                grad[2*j+1] += w * (-d[1]) / dist
            idx += 1
    
    return -soft_min_val, grad


def _optimize_lbfgs(initial_points, max_iter=200, beta=100.0):
    """Run L-BFGS-B with analytic gradient from initial points."""
    def obj_and_grad(x):
        val, grad = _objective_and_gradient(x, beta)
        return val, grad
    
    result = minimize(
        obj_and_grad,
        initial_points.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    best_points = result.x.reshape(N, 2)
    # Clip to unit disk
    r = np.sqrt(np.sum(best_points**2, axis=1))
    mask = r > 1.0
    if np.any(mask):
        best_points[mask] /= r[mask, np.newaxis]
    best_score = _score(best_points)
    return best_points, best_score


def _get_hex_lattice():
    """Generate initial points on a hexagonal lattice inside the unit disk."""
    points = []
    spacing = 0.15
    while len(points) < N:
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= (1.0 - spacing/2.0)**2:
                    points.append([x, y])
        if len(points) >= N:
            break
        spacing *= 0.92
    points = np.array(points)
    radii = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(radii)
    return points[idx[:N]]


def _get_ring_init():
    """Generate points on concentric rings."""
    points = []
    points.append([0.0, 0.0])
    n_rings = 7
    for ring in range(1, n_rings):
        r = ring * 0.12
        n_pts = min(6 * ring, N - len(points))
        if n_pts <= 0:
            break
        angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    points = np.array(points[:N])
    return points


def _get_random_init():
    """Generate random initial points inside the unit disk."""
    angles = np.random.uniform(0, 2*np.pi, N)
    radii = np.random.uniform(0, 0.95, N)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def _dynamics_simulate(initial_points, dt=0.001, n_steps=5000, damping=0.99, 
                       repulsion_strength=10.0, boundary_strength=50.0,
                       time_limit=None):
    """
    Rigid-body dynamics simulation with:
    - Mutual repulsion between circles (inverse-square or linear)
    - Boundary repulsion to keep circles inside unit disk
    - Velocity damping for settling
    - Symplectic (velocity Verlet) integration
    """
    if time_limit is None:
        time_limit = 20.0
    
    start_time = time.time()
    
    positions = initial_points.copy().astype(float)
    velocities = np.zeros_like(positions)
    
    # Estimate a reasonable repulsion radius
    # We want circles to be separated; use a target spacing
    target_spacing = 0.15
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                d = positions[i] - positions[j]
                dist = np.sqrt(np.sum(d**2))
                if dist < 1e-10:
                    # Random nudge
                    d = np.random.uniform(-0.01, 0.01, 2)
                    dist = np.sqrt(np.sum(d**2))
                
                # Repulsive force: strong when close, decays with distance
                if dist < target_spacing * 2:
                    # Linear repulsion scaled to be strong at contact
                    force_mag = repulsion_strength * (target_spacing - dist) / target_spacing
                    if force_mag > 0:
                        force_vec = force_mag * d / dist
                        forces[i] += force_vec
                        forces[j] -= force_vec
                else:
                    # Weaker long-range repulsion to maintain spacing
                    force_mag = repulsion_strength * 0.1 * (target_spacing - dist) / (dist + 1e-6)
                    force_vec = force_mag * d / dist
                    forces[i] += force_vec
                    forces[j] -= force_vec
        
        # Boundary force: push circles inward if they approach the unit circle
        for i in range(N):
            r_i = np.sqrt(np.sum(positions[i]**2))
            if r_i > 1e-10:
                # Force proportional to how close to boundary
                margin = 1.0 - r_i
                if margin < target_spacing:
                    force_mag = boundary_strength * (target_spacing - margin) / target_spacing
                    if force_mag > 0:
                        # Direction: toward center
                        force_vec = force_mag * (-positions[i] / r_i)
                        forces[i] += force_vec
        
        # Velocity Verlet integration
        # Update velocities
        velocities += forces * dt
        velocities *= damping
        
        # Update positions
        positions += velocities * dt
        
        # Hard constraint: clamp to unit disk
        r = np.sqrt(np.sum(positions**2, axis=1))
        mask = r > 1.0
        if np.any(mask):
            positions[mask] /= r[mask, np.newaxis]
            # Reflect velocity
            for i in np.where(mask)[0]:
                r_i = positions[i] / np.sqrt(np.sum(positions[i]**2))
                v_dot_r = np.dot(velocities[i], r_i)
                if v_dot_r > 0:
                    velocities[i] -= 2 * v_dot_r * r_i
        
    return positions


def _polish(points, max_iter=300):
    """Polish with L-BFGS-B using adaptive beta."""
    # Start with moderate beta, increase for final refinement
    best_points = points
    best_score = _score(points)
    
    for beta in [50.0, 100.0, 200.0, 500.0]:
        polished, score = _optimize_lbfgs(points, max_iter=max_iter, beta=beta)
        if score > best_score:
            best_score = score
            best_points = polished
        points = polished
    
    return best_points, best_score


def _high_beta_polish(points, max_iter=500):
    """Final high-precision polish with very high beta."""
    polished, score = _optimize_lbfgs(points, max_iter=max_iter, beta=1000.0)
    if score < _score(points):
        return points, _score(points)
    return polished, score


def construct():
    start_time = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Dynamics-based initialization from multiple starts
    initializers = [
        _get_hex_lattice,
        _get_ring_init,
        _get_random_init,
        _get_random_init,
    ]
    
    for init_func in initializers:
        if time.time() - start_time > DEADLINE_SECONDS - 30:
            break
        
        initial = init_func()
        
        # Run dynamics simulation
        sim_points = _dynamics_simulate(
            initial,
            dt=0.0005,
            n_steps=3000,
            damping=0.995,
            repulsion_strength=15.0,
            boundary_strength=80.0,
            time_limit=8.0
        )
        
        # Polish the dynamics result
        polished, score = _polish(sim_points, max_iter=200)
        
        if score > best_score:
            best_score = score
            best_points = polished
    
    # Phase 2: If we have time, do additional random restarts with dynamics
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - 15 and best_points is not None:
        remaining = DEADLINE_SECONDS - 15 - elapsed
        # Try a few more random starts
        n_extra = min(int(remaining / 10), 5)
        for _ in range(n_extra):
            if time.time() - start_time > DEADLINE_SECONDS - 10:
                break
            initial = _get_random_init()
            sim_points = _dynamics_simulate(
                initial,
                dt=0.0005,
                n_steps=2000,
                damping=0.995,
                repulsion_strength=15.0,
                boundary_strength=80.0,
                time_limit=5.0
            )
            polished, score = _polish(sim_points, max_iter=150)
            if score > best_score:
                best_score = score
                best_points = polished
    
    # Phase 3: Final high-beta polish
    if best_points is not None and time.time() - start_time < DEADLINE_SECONDS - 5:
        final_points, final_score = _high_beta_polish(best_points, max_iter=300)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
    
    if best_points is None:
        best_points = _get_hex_lattice()
    
    return best_points
