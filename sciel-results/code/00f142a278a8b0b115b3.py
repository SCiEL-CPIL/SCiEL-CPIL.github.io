"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START) - 2.0  # 2s safety margin

def _fast_energy(pts):
    """Fast logarithmic energy using NumPy sum. Not math.fsum, for speed during optimization."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    dsq = dist_sq[iu]
    dsq = np.maximum(dsq, 1e-30)
    # log(1/sqrt(dsq)) = -0.5 * log(dsq)
    return -0.5 * np.sum(np.log(dsq))

def _fast_grad(pts):
    """Gradient of log(1/d) energy w.r.t. each point (in R^3). Vectorized."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-30)
    inv_dsq = 1.0 / dist_sq
    inv_dsq[np.diag_indices(n)] = 0.0
    # grad[i] = -sum_j (x_i - x_j) / |x_i - x_j|^2
    grad = -np.sum(diff * inv_dsq[:, :, np.newaxis], axis=1)
    return grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _sphere_fun_grad(x_flat):
    """Objective and gradient for L-BFGS on flat vector; projects to sphere."""
    pts = _project_to_sphere(x_flat.reshape(N, 3))
    e = _fast_energy(pts)
    g = _fast_grad(pts)
    # Project gradient onto tangent space: g_tangent = g - (g.x)x
    dot = np.sum(pts * g, axis=1, keepdims=True)
    g_tangent = g - dot * pts
    return e, g_tangent.flatten()

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _lbfgs_optimize(pts, max_iter=500, gtol=1e-10):
    """Run L-BFGS-B on the sphere using projected gradients."""
    x0 = pts.flatten()
    best_e = _fast_energy(pts)
    best_pts = pts.copy()
    
    def fun_grad(x):
        e, g = _sphere_fun_grad(x)
        return e, g

    try:
        res = minimize(fun_grad, x0, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'gtol': gtol,
                               'maxfun': max_iter * 3})
        if res.fun < best_e:
            new_pts = _project_to_sphere(res.x.reshape(N, 3))
            new_e = _fast_energy(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_pts = new_pts
    except Exception:
        pass
    return best_pts, best_e

def _kick_and_optimize(pts, num_kicks=5, kick_size=0.2):
    """Perturb a few points and re-optimize to escape local minima."""
    best_pts = pts.copy()
    best_e = _fast_energy(pts)
    rng = np.random.RandomState(12345)
    
    for _ in range(num_kicks):
        if _time_left() < 10.0:
            break
        # Pick 5-10 random points to perturb
        n_perturb = rng.randint(5, 11)
        idx = rng.choice(N, n_perturb, replace=False)
        new_pts = pts.copy()
        for i in idx:
            noise = rng.randn(3) * kick_size
            new_pts[i] = _project_to_sphere((pts[i] + noise).reshape(1, 3))[0]
        
        # Optimize the perturbed configuration
        opt_pts, opt_e = _lbfgs_optimize(new_pts, max_iter=200, gtol=1e-8)
        
        if opt_e < best_e:
            best_e = opt_e
            best_pts = opt_pts
            pts = best_pts # Update base for next kick
    
    return best_pts, best_e

def construct():
    global _START
    _START = time.time()
    
    best_pts = None
    best_e = float('inf')
    
    # Phase 1: Multiple diverse starts
    # Start 1: Fibonacci
    if _time_left() > 10.0:
        pts = _fibonacci_spiral()
        pts, e = _lbfgs_optimize(pts, max_iter=300, gtol=1e-8)
        if e < best_e:
            best_e = e
            best_pts = pts
    
    # Start 2: Random
    if _time_left() > 10.0:
        pts = _random_points(seed=1)
        pts, e = _lbfgs_optimize(pts, max_iter=300, gtol=1e-8)
        if e < best_e:
            best_e = e
            best_pts = pts
            
    # Start 3: Random (different seed)
    if _time_left() > 10.0:
        pts = _random_points(seed=42)
        pts, e = _lbfgs_optimize(pts, max_iter=300, gtol=1e-8)
        if e < best_e:
            best_e = e
            best_pts = pts

    # Phase 2: Kick and optimize to escape local minima
    if _time_left() > 20.0 and best_pts is not None:
        best_pts, best_e = _kick_and_optimize(best_pts, num_kicks=10, kick_size=0.3)
        
    # Phase 3: Further kicks with smaller size if time permits
    if _time_left() > 15.0 and best_pts is not None:
        best_pts, best_e = _kick_and_optimize(best_pts, num_kicks=10, kick_size=0.1)

    # Phase 4: Final high-precision polish
    if _time_left() > 5.0 and best_pts is not None:
        best_pts, best_e = _lbfgs_optimize(best_pts, max_iter=1000, gtol=1e-12)
        
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_spiral()
        
    best_pts = _project_to_sphere(best_pts)
    
    # Final validation: check for close points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                rng = np.random.RandomState(999)
                best_pts[j] = _project_to_sphere(best_pts[j] + 1e-8 * rng.randn(3))[0]
                
    return best_pts
# EVOLVE-BLOCK-END
