"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy using math.fsum for accuracy."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    energy = _log_energy(points)
    return energy, grads_tangent.flatten()


def _lbfgs_optimize(x0, maxiter=5000, ftol=1e-14, gtol=1e-12, maxfun=50000):
    """Run L-BFGS-B optimization and return (energy, points)."""
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxfun, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return e_check, p_res
    except Exception:
        return float('inf'), None


def _basin_hopping_step(points, step_size=0.1, n_perturb=5, rng=None):
    """
    Perform basin-hopping: perturb points, then locally optimize.
    Returns the best (energy, points) found.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    best_e = _log_energy(points)
    best_p = points.copy()
    
    for _ in range(n_perturb):
        # Add random perturbation
        noise = rng.randn(N, 3) * step_size
        perturbed = _project_to_sphere(points + noise)
        
        # Local optimize
        e, p = _lbfgs_optimize(perturbed.flatten(), maxiter=2000, ftol=1e-12, gtol=1e-9, maxfun=20000)
        
        if e < best_e:
            best_e = e
            best_p = p
    
    return best_e, best_p


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Multiple initializations with diversity
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=4),
        _random_uniform_sphere(N, seed=5),
        _random_uniform_sphere(N, seed=6),
        _random_uniform_sphere(N, seed=7),
        _random_uniform_sphere(N, seed=8),
    ]
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # Phase 1: L-BFGS from each initialization
    for idx, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        x0 = init.flatten()
        e, p = _lbfgs_optimize(x0, maxiter=3000, ftol=1e-12, gtol=1e-9, maxfun=30000)
        
        if e < best_energy:
            best_energy = e
            best_points = p.copy()
    
    # Phase 2: Basin hopping around the best solution
    rng = np.random.RandomState(123)
    for step_size in [0.3, 0.2, 0.1, 0.05, 0.02]:
        if time.time() > deadline - 15.0:
            break
        
        e, p = _basin_hopping_step(best_points, step_size=step_size, n_perturb=5, rng=rng)
        if e < best_energy:
            best_energy = e
            best_points = p.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5.0:
        x_best = best_points.flatten()
        e, p = _lbfgs_optimize(x_best, maxiter=8000, ftol=1e-15, gtol=1e-13, maxfun=80000)
        if e < best_energy:
            best_energy = e
            best_points = p
    
    # Phase 4: Second polish pass if time allows
    if time.time() < deadline - 2.0:
        x_best = best_points.flatten()
        e, p = _lbfgs_optimize(x_best, maxiter=5000, ftol=1e-15, gtol=1e-14, maxfun=50000)
        if e < best_energy:
            best_energy = e
            best_points = p
    
    best_points = _project_to_sphere(best_points)
    
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
