"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on the unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|)
    Gradient w.r.t. x_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    N = X.shape[0]
    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle only
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j!=i} (x_i - x_j) / dist_ij^2
    # = sum_{j!=i} diff[i,j] / dist_sq[i,j]
    # diff[i,j] = x_i - x_j, so (x_i - x_j)/dist_sq[i,j]
    grad = np.zeros_like(X)
    for i in range(N):
        # Sum over j != i of (x_i - x_j) / dist_sq[i,j]
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        # diff[i, j] = x_i - x_j
        contrib = diff[i, mask] / dist_sq[i, mask][:, np.newaxis]
        grad[i] = np.sum(contrib, axis=0)
    
    return energy, grad


def _project_to_sphere(X):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return X / norms


def _initial_fibonacci(seed=0):
    """Generate Fibonacci spiral points with optional perturbation."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    if seed > 0:
        rng = np.random.default_rng(seed)
        X = X + 0.1 * rng.standard_normal((N, 3))
        X = _project_to_sphere(X)
    
    return X


def _optimize_lbfgs(X0, time_limit=100.0):
    """Optimize using L-BFGS-B with deadline guard."""
    start_time = time.time()
    
    # Flatten for scipy
    x0 = X0.flatten()
    
    def objective(x):
        X = x.reshape(N, 3)
        X = _project_to_sphere(X)
        e, g = _energy_and_grad(X)
        return e
    
    def gradient(x):
        X = x.reshape(N, 3)
        X = _project_to_sphere(X)
        e, g = _energy_and_grad(X)
        return g.flatten()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        return _project_to_sphere(result.x.reshape(N, 3)), result.fun
    except Exception:
        return X0, _energy_and_grad(X0)[0]


def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin before 150s kill
    
    # Start with Fibonacci spiral
    X_best = _initial_fibonacci(0)
    e_best = _energy_and_grad(X_best)[0]
    
    # Try multiple restarts with different perturbations
    restarts = [(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.15), (4, 0.25)]
    
    for seed, perturbation in restarts:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        X_start = _initial_fibonacci(seed) if seed > 0 else X_best.copy()
        if seed > 0:
            rng = np.random.default_rng(seed)
            X_start = X_start + perturbation * rng.standard_normal((N, 3))
            X_start = _project_to_sphere(X_start)
        
        X_opt, e_opt = _optimize_lbfgs(X_start, time_limit=120.0)
        
        if e_opt < e_best:
            e_best = e_opt
            X_best = X_opt
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
    
    # Final validation
    X_best = _project_to_sphere(X_best)
    
    # Check for very close points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(X_best[i] - X_best[j])
            if d < 1e-9:
                # Perturb slightly
                rng = np.random.default_rng(42 + i + j)
                X_best[j] = X_best[j] + 1e-6 * rng.standard_normal(3)
                X_best[j] /= np.linalg.norm(X_best[j])
    
    return X_best
# EVOLVE-BLOCK-END
