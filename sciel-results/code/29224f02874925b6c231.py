import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists_sq = np.sum(diff * diff, axis=-1)
    dists = np.sqrt(np.maximum(dists_sq, 1e-30))
    log_dists = np.log(dists)
    return -math.fsum(log_dists.tolist())

def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1)
    weighted_points = inv_dists_sq @ points
    
    grad = weighted_points - points * sum_inv_dists_sq[:, np.newaxis]
    
    dot_products = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - dot_products * points
    
    # Calculate energy efficiently
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(np.maximum(dists_sq[iu], 1e-30))
    energy = -math.fsum(np.log(dists).tolist())
    
    return energy, grad_tangent.flatten()

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, rng):
    """Generate random uniform points on sphere."""
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _md_init(n, steps=2000, dt=0.01, cooling=0.99, seed=42):
    """
    Initialize points using simple Molecular Dynamics with repulsive forces.
    Force F_ij = (x_i - x_j) / |x_i - x_j|^3 (inverse square law for repulsion)
    This mimics the gradient of -log(d) roughly, but is faster to converge to a good config.
    """
    rng = np.random.RandomState(seed)
    points = _random_uniform_sphere(n, rng)
    velocities = np.zeros_like(points)
    
    for step in range(steps):
        # Compute forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        # Force is proportional to diff / dist^3
        # To avoid singularity, we use a softened potential or just clip distances
        inv_dists_cubed = 1.0 / (dists_sq * np.sqrt(dists_sq))
        np.fill_diagonal(inv_dists_cubed, 0.0)
        
        forces = np.sum(diff * inv_dists_cubed[:, :, np.newaxis], axis=1)
        
        # Update velocities (simple Euler with damping)
        velocities = velocities * cooling + forces * dt
        points = points + velocities * dt
        
        # Project to sphere
        points = _project_to_sphere(points)
        # Zero out radial component of velocity to keep them on sphere
        radial_vel = np.sum(velocities * points, axis=1, keepdims=True)
        velocities = velocities - radial_vel * points
        
    return points

def _local_minimize(points, deadline, maxiter=500, tol=1e-14):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': tol, 'gtol': tol, 'maxls': 40}
        )
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)

def _basin_hopping(init_points, deadline, rng, max_iter=100):
    """Basin hopping with L-BFGS-B local minimization."""
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=200)
    best_points = current_points.copy()
    best_energy = current_energy
    
    T = 0.1
    T_min = 1e-6
    cooling = 0.97
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        if T < T_min:
            break
            
        # Perturb
        # Randomly choose a few points to move
        n_perturb = rng.randint(2, 10)
        idxs = rng.choice(N, n_perturb, replace=False)
        
        new_points = current_points.copy()
        for idx in idxs:
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, new_points[idx]) * new_points[idx]
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-15:
                tangent /= norm_t
            else:
                continue
            step_size = T * rng.uniform(0.5, 1.5)
            new_pt = new_points[idx] + step_size * tangent
            new_points[idx] = new_pt / np.linalg.norm(new_pt)
        
        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=200)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
        
        T *= cooling

    return best_points, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random points
    for seed in range(3):
        r = np.random.RandomState(seed)
        inits.append(_random_uniform_sphere(N, r))
        
    # 3. MD initialized points (usually very good starting points)
    for seed in range(3):
        md_pts = _md_init(N, steps=1000, seed=seed)
        inits.append(md_pts)

    # Best so far
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # Phase 1: Local minimize all inits
    local_min_results = []
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        p_res, e_res = _local_minimize(init, deadline, maxiter=500)
        local_min_results.append((e_res, p_res))
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()

    # Sort local minima by energy
    local_min_results.sort(key=lambda x: x[0])
    
    # Phase 2: Basin hopping from the best candidates
    # We run BH from the top 3 distinct local minima
    num_bh_runs = min(3, len(local_min_results))
    for i in range(num_bh_runs):
        if time.time() > deadline - 15.0:
            break
        bh_rng = np.random.RandomState(i * 100 + 123)
        bh_points, bh_energy = _basin_hopping(local_min_results[i][1], deadline, bh_rng, max_iter=50)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_points = bh_points.copy()

    # Phase 3: Final polishing with very high precision
    if time.time() < deadline - 2.0:
        final_points, final_energy = _local_minimize(best_points, deadline, maxiter=2000, tol=1e-15)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points.copy()

    best_points = _project_to_sphere(best_points)

    # Safety checks
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)

    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)

    return best_points
