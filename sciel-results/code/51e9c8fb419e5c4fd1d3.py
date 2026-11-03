"""Rewritten program for logarithmic energy minimization of 150 points on sphere.
Uses a hybrid strategy: Fast Riemannian Gradient Descent for exploration 
and L-BFGS-B for high-precision exploitation.
"""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    # Compute pairwise distances
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists_sq = np.sum(diff * diff, axis=-1)
    # Clip to avoid log(0) or log(negative due to floating point)
    dists = np.sqrt(np.maximum(dists_sq, 1e-30))
    log_dists = np.log(dists)
    return -math.fsum(log_dists.tolist())

def _compute_gradient(points):
    """Compute the Euclidean gradient of the energy.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    """
    n = len(points)
    # diff[i, j, :] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero
    
    # weights[i, j] = 1 / d_ij^2
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # grad[i] = - sum_j inv_dists_sq[i, j] * diff[i, j]
    # diff[i, j] = points[i] - points[j]
    # So grad[i] = - sum_j (points[i] - points[j]) / d_ij^2
    #            = - points[i] * sum_j (1/d_ij^2) + sum_j (points[j] / d_ij^2)
    
    sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1)
    weighted_points = inv_dists_sq @ points
    
    grad = weighted_points - points * sum_inv_dists_sq[:, np.newaxis]
    return grad

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _riemannian_gradient_descent(points, max_iter=5000, tol=1e-10, deadline=None):
    """
    Fast Riemannian Gradient Descent with Momentum.
    This is significantly faster than L-BFGS-B for the initial exploration phase.
    """
    points = points.copy()
    current_energy = _log_energy(points)
    best_energy = current_energy
    best_points = points.copy()
    
    # Momentum parameters
    beta = 0.9
    lr = 0.05
    momentum = np.zeros_like(points)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute gradient
        grad = _compute_gradient(points)
        
        # Project to tangent space
        dot_products = np.sum(grad * points, axis=1, keepdims=True)
        grad_tangent = grad - dot_products * points
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
            
        # Update momentum
        momentum = beta * momentum - lr * grad_tangent
        
        # Update points
        new_points = points + momentum
        new_points = _project_to_sphere(new_points)
        
        # Compute new energy
        new_energy = _log_energy(new_points)
        
        # Adaptive step size / Backtracking
        if new_energy < current_energy:
            current_energy = new_energy
            points = new_points
            # Slowly decrease learning rate
            lr *= 0.999
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        else:
            # Reduce learning rate and revert
            lr *= 0.5
            # Revert momentum to avoid sticking in bad direction
            momentum = beta * momentum
            # Don't update points
            # If lr gets too small, break
            if lr < 1e-8:
                break
                
    return best_points, best_energy

def _local_minimize_lbfgs(points, deadline, maxiter=1000, tol=1e-14):
    """Run L-BFGS-B to find local minimum. Used for polishing."""
    
    def _energy_and_grad(x):
        pts = x.reshape(N, 3)
        pts = _project_to_sphere(pts)
        energy = _log_energy(pts)
        grad = _compute_gradient(pts)
        # Project gradient to tangent space
        dot_products = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tangent = grad - dot_products * pts
        return energy, grad_tangent.flatten()

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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random points (more diverse than before)
    for seed in range(10):
        r = np.random.RandomState(seed + 100)
        inits.append(_random_uniform_sphere(N, r))
        
    # 3. Perturbed Fibonacci (various noise levels)
    for k in range(5):
        noise_level = 0.05 * (k + 1)
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * noise_level
        inits.append(_project_to_sphere(perturbed))
        
    # 4. Clustered inits to test repulsion
    base = _fibonacci_sphere(N // 2)
    cluster1 = base + rng.randn(N // 2, 3) * 0.1
    cluster2 = -base + rng.randn(N // 2, 3) * 0.1
    inits.append(_project_to_sphere(np.vstack([cluster1, cluster2])))

    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # Phase 1: Fast Riemannian Gradient Descent from all inits
    # This is the structural change: using fast GD for exploration
    gd_results = []
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        p_res, e_res = _riemannian_gradient_descent(init, max_iter=2000, deadline=deadline - 20.0)
        gd_results.append((e_res, p_res))
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()
            
    # Sort GD results to get the best candidates for polishing
    gd_results.sort(key=lambda x: x[0])
    
    # Phase 2: Polish the top candidates with L-BFGS-B
    # We only polish the top 3-5 to save time for more GD runs if needed
    top_k = min(5, len(gd_results))
    polished_results = []
    
    for i in range(top_k):
        if time.time() > deadline - 5.0:
            break
        e_init, p_init = gd_results[i]
        p_res, e_res = _local_minimize_lbfgs(p_init, deadline, maxiter=500, tol=1e-12)
        polished_results.append((e_res, p_res))
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()
            
    # Phase 3: If time remains, try more GD runs from different random seeds
    # This leverages the speed of GD to explore more basins
    extra_seed = 200
    while time.time() < deadline - 10.0:
        r = np.random.RandomState(extra_seed)
        init = _random_uniform_sphere(N, r)
        p_res, e_res = _riemannian_gradient_descent(init, max_iter=1500, deadline=deadline - 10.0)
        if e_res < best_energy:
            # If GD finds a better spot, polish it immediately
            p_res2, e_res2 = _local_minimize_lbfgs(p_res, deadline, maxiter=200, tol=1e-12)
            if e_res2 < best_energy:
                best_energy = e_res2
                best_points = p_res2.copy()
        extra_seed += 1
        
    # Phase 4: Final high-precision polish on the absolute best
    if time.time() < deadline - 2.0:
        final_points, final_energy = _local_minimize_lbfgs(best_points, deadline, maxiter=2000, tol=1e-14)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points.copy()

    best_points = _project_to_sphere(best_points)

    # Safety checks
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)

    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)

    return best_points
# EVOLVE-BLOCK-END
