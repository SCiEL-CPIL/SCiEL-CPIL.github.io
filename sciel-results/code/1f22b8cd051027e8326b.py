"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi-xj|) and its gradient.
    pts: (N, 3) array of points on unit sphere (approximately).
    Returns (energy, grad) where grad is (N, 3).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Pairwise distances
    # d[i,j] = |xi - xj|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    
    # Filter out very small distances to avoid log(0)
    valid = d_upper > 1e-12
    d_valid = d_upper[valid]
    
    energy = -np.sum(np.log(d_valid))
    
    # Gradient computation:
    # d/dxi [log(1/|xi-xj|)] = d/dxi [-log(|xi-xj|)] = -(xi-xj)/|xi-xj|^2
    # For point i, grad_i = sum_{j != i} -(xi - xj) / |xi - xj|^2
    #                     = sum_{j != i} (xj - xi) / |xi - xj|^2
    
    grad = np.zeros_like(pts)
    # For each pair (i,j), contribution to grad_i is (xj - xi)/d^2, to grad_j is (xi - xj)/d^2
    # Using the diff array: diff[i,j] = xi - xj
    # So (xj - xi) = -diff[i,j]
    # Contribution to grad_i from pair (i,j): -diff[i,j] / d^2
    # Contribution to grad_j from pair (i,j): diff[i,j] / d^2
    
    # We only need upper triangle pairs
    # For pair (i,j) with i<j:
    #   grad[i] += -diff[i,j] / dists[i,j]^2
    #   grad[j] += diff[i,j] / dists[i,j]^2
    
    # Efficient vectorized computation
    # diff has shape (n, n, 3), dists has shape (n, n)
    # inv_d2 = 1 / dists^2, but set diagonal to 0
    inv_d2 = np.zeros_like(dists)
    np.fill_diagonal(inv_d2, 0.0)
    # Only compute where dists is reasonable
    mask = dists > 1e-12
    inv_d2[mask] = 1.0 / (dists[mask] * dists[mask])
    
    # grad_i = sum_j -diff[i,j] * inv_d2[i,j]
    # = -sum_j diff[i,j] * inv_d2[i,j]
    # diff is (n, n, 3), inv_d2 is (n, n)
    # Multiply: (n, n, 3) * (n, n, 1) -> sum over j (axis=1)
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _energy_only(pts):
    """Energy only, for evaluation."""
    e, _ = _energy_and_grad(pts)
    return e


def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    Returns array of shape (100, 3).
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_pts = None
    best_energy = np.inf
    
    def _evaluate(pts):
        nonlocal best_pts, best_energy
        pts = _project_to_sphere(pts)
        # Check validity
        if not np.all(np.isfinite(pts)):
            return
        # Check min distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d < 1e-9:
            return
        e, _ = _energy_and_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
        return e
    
    # Initial configurations
    configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    configs.append(fib)
    
    # 2. Random points (multiple seeds)
    rng = np.random.default_rng(42)
    for seed in range(5):
        rng_seed = np.random.default_rng(42 + seed)
        pts = rng_seed.standard_normal((N, 3))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
    
    # 3. Jittered Fibonacci
    for jitter_scale in [0.01, 0.05, 0.1]:
        rng_j = np.random.default_rng(100)
        jitter = rng_j.standard_normal((N, 3)) * jitter_scale
        jittered = fib + jitter
        jittered = _project_to_sphere(jittered)
        configs.append(jittered)
    
    # 4. Two-hemisphere random
    rng_h = np.random.default_rng(200)
    for _ in range(3):
        pts = rng_h.standard_normal((N, 3))
        # Bias toward poles
        pts[:, 2] = np.abs(pts[:, 2]) ** 0.5 * np.sign(pts[:, 2])
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
    
    # Optimize each configuration
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline:
            break
        
        init_pts = _project_to_sphere(init_pts)
        
        # First pass: L-BFGS-B with moderate tolerance
        try:
            res = minimize(
                fun=lambda x: _energy_and_grad(x)[0],
                x0=init_pts.flatten(),
                jac=lambda x: _energy_and_grad(x)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            if res.success or res.fun < best_energy:
                opt_pts = _project_to_sphere(res.x.reshape(N, 3))
                _evaluate(opt_pts)
        except Exception:
            pass
        
        # Second pass: tighter tolerance from best result
        if time.time() > deadline:
            break
        
        if best_pts is not None:
            try:
                res2 = minimize(
                    fun=lambda x: _energy_and_grad(x)[0],
                    x0=best_pts.flatten(),
                    jac=lambda x: _energy_and_grad(x)[1].flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                if res2.success or res2.fun < best_energy:
                    opt_pts2 = _project_to_sphere(res2.x.reshape(N, 3))
                    _evaluate(opt_pts2)
            except Exception:
                pass
        
        # If time remains, do a final polish on the best
        if time.time() > deadline - 5:
            break
        
        if best_pts is not None:
            try:
                res3 = minimize(
                    fun=lambda x: _energy_and_grad(x)[0],
                    x0=best_pts.flatten(),
                    jac=lambda x: _energy_and_grad(x)[1].flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-16}
                )
                if res3.success or res3.fun < best_energy:
                    opt_pts3 = _project_to_sphere(res3.x.reshape(N, 3))
                    _evaluate(opt_pts3)
            except Exception:
                pass
    
    # Final: if we have a best, do one more tight polish
    if time.time() < deadline - 2 and best_pts is not None:
        try:
            res_final = minimize(
                fun=lambda x: _energy_and_grad(x)[0],
                x0=best_pts.flatten(),
                jac=lambda x: _energy_and_grad(x)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 3000, 'gtol': 1e-13, 'ftol': 1e-16}
            )
            if res_final.success or res_final.fun < best_energy:
                opt_final = _project_to_sphere(res_final.x.reshape(N, 3))
                _evaluate(opt_final)
        except Exception:
            pass
    
    # Ensure we return something valid
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = fib
    
    # Final normalization and validation
    best_pts = _project_to_sphere(best_pts)
    if not np.all(np.isfinite(best_pts)):
        best_pts = fib
    
    return best_pts
# EVOLVE-BLOCK-END
