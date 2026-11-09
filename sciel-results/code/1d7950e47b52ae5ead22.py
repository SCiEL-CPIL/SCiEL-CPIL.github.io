import numpy as np
import time
from scipy.optimize import minimize
import math

N = 100
DEADLINE = 110.0  # safety margin before 120s limit

def _energy_and_gradient(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient wrt x_i (in R^3): -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    # Compute all pairwise distances
    # pts[i] - pts[j] for all i, j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist[i,j])
    # Only use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # grad[i] = -sum_j diff[i,j] / dist[i,j]^2
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[~np.eye(n, dtype=bool)] = 1.0 / (dist[~np.eye(n, dtype=bool)] ** 2)
    
    # grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    """Normalize each point to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_initialization():
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_initialization(seed=42):
    """Random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _differential_evolution_init():
    """Use differential evolution to find a good initial configuration."""
    from scipy.optimize import differential_evolution
    
    def objective_flat(flat):
        pts = flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_gradient(pts)
        return e
    
    bounds = [(-1, 1)] * (N * 3)
    
    try:
        result = differential_evolution(
            objective_flat,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=123,
            polish=False,
            updating='immediate'
        )
        best_pts = result.x.reshape(N, 3)
        best_pts = _project_to_sphere(best_pts)
        return best_pts
    except Exception:
        return _random_initialization(seed=99)

def _lbfgs_refine(pts0, maxiter=500, gtol=1e-10):
    """Refine points using L-BFGS-B with sphere projection."""
    def objective_and_grad(flat):
        pts = flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _energy_and_gradient(pts)
        # Project gradient to tangent space
        # g_tangent = g - (g . x_i) * x_i for each i
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_tangent.flatten()
    
    flat0 = pts0.flatten()
    
    result = minimize(
        objective_and_grad,
        flat0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-14}
    )
    
    best_pts = result.x.reshape(N, 3)
    best_pts = _project_to_sphere(best_pts)
    return best_pts, result.fun

def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_initialization())
    
    # 2. Several random initializations
    for seed in [42, 123, 456, 789, 101112]:
        inits.append(_random_initialization(seed=seed))
    
    # 3. Differential evolution initialization (if time permits)
    elapsed = time.time() - start_time
    if elapsed < 30.0:
        de_init = _differential_evolution_init()
        inits.append(de_init)
    
    best_energy = np.inf
    best_pts = inits[0]
    
    for idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break
        
        # Refine with L-BFGS
        try:
            refined_pts, energy = _lbfgs_refine(init_pts, maxiter=300, gtol=1e-10)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_pts = refined_pts
        except Exception:
            pass
    
    # Final high-precision polish on the best result
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5.0:
        try:
            polished_pts, energy = _lbfgs_refine(best_pts, maxiter=1000, gtol=1e-12)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_pts = polished_pts
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if points are too close
        best_pts = _fibonacci_initialization()
    
    return best_pts
