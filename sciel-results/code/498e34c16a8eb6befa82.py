"""Optimized placement of 14 points in 3D to minimize squared max/min distance ratio."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import itertools
import time

_START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _pairwise_distances(pts):
    """Compute all pairwise squared distances for an (n, d) array."""
    n = pts.shape[0]
    # (xi - xj)^2 = |xi|^2 + |xj|^2 - 2 xi.xj
    sq_norms = np.sum(pts**2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * (pts @ pts.T)
    np.fill_diagonal(sq_dists, 0.0)
    return sq_dists

def _score_from_sq_dists(sq_dists):
    """Compute (max_d / min_d)^2 from squared distances matrix."""
    # Only consider upper triangle (i < j) for distinct pairs
    iu = np.triu_indices(sq_dists.shape[0], k=1)
    d2 = sq_dists[iu]
    d_min = np.min(d2)
    d_max = np.max(d2)
    if d_min <= 0:
        return np.inf
    return (d_max / d_min)

def _objective(params):
    """Objective: minimize squared ratio. params shape (n*d,)."""
    pts = params.reshape(N, D)
    sq_dists = _pairwise_distances(pts)
    return _score_from_sq_dists(sq_dists)

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    best_score = np.inf
    best_pts = None
    
    # Strategy 1: Try stella octangula (two interlocking tetrahedra)
    # Regular tetrahedron vertices (unit edge length)
    # Using standard regular tetrahedron coordinates
    # Vertices of regular tetrahedron with edge length 1:
    # (1,1,1)/sqrt(3), (1,-1,-1)/sqrt(3), (-1,1,-1)/sqrt(3), (-1,-1,1)/sqrt(3)
    # Actually let's use a cleaner parameterization
    
    # Regular tetrahedron with edge length s:
    # v1 = (1, 1, 1) * s / (2*sqrt(2))
    # v2 = (1, -1, -1) * s / (2*sqrt(2))
    # v3 = (-1, 1, -1) * s / (2*sqrt(2))
    # v4 = (-1, -1, 1) * s / (2*sqrt(2))
    # Edge length between any two: sqrt((2)^2 + (2)^2 + (0)^2) * s/(2*sqrt(2)) = sqrt(8)*s/(2*sqrt(2)) = 2*sqrt(2)*s/(2*sqrt(2)) = s
    
    # For stella octangula: two tetrahedra, one rotated 180° about an axis
    # The compound has 8 vertices total. But we need 14 points.
    
    # Alternative: Use a known good starting configuration and optimize
    # Start with points on a sphere or known good arrangement
    
    # Let's try a different approach: start with points distributed on a sphere
    # and use differential evolution on the full 42-dimensional space
    
    # But 42D is high. Let's try a hybrid approach.
    
    # First, try the stella octangula idea with 8 points, then add 6 more
    # Actually, let's just use differential evolution with a good initial bound
    
    # Better: use multiple strategies
    # Strategy 1: Differential evolution in 42D with reasonable bounds
    # Strategy 2: Local optimization from good starting points
    
    # Let's use a sphere-based initialization
    # Points roughly on a sphere of radius 1
    
    # Fibonacci sphere for 14 points
    def fibonacci_sphere(n):
        pts = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pts.append([x, y, z])
        return np.array(pts)
    
    # Initial guess: Fibonacci sphere
    init_pts = fibonacci_sphere(N)
    
    # Try differential evolution in 42D
    if _time_remaining() > 60:
        bounds = [(-2, 2)] * (N * D)
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=False,
                callback=None
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_pts = de_result.x.reshape(N, D)
        except Exception:
            pass
    
    # Local refinement using L-BFGS-B
    if best_pts is not None and _time_remaining() > 20:
        try:
            local_result = minimize(
                _objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if local_result.fun < best_score:
                best_score = local_result.fun
                best_pts = local_result.x.reshape(N, D)
        except Exception:
            pass
    
    # Also try from Fibonacci sphere directly with local optimization
    if _time_remaining() > 20:
        try:
            local_result2 = minimize(
                _objective,
                init_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if local_result2.fun < best_score:
                best_score = local_result2.fun
                best_pts = local_result2.x.reshape(N, D)
        except Exception:
            pass
    
    # Try a few random restarts with local optimization
    np.random.seed(123)
    for _ in range(5):
        if _time_remaining() < 10:
            break
        random_pts = np.random.uniform(-1, 1, size=(N, D))
        try:
            local_result3 = minimize(
                _objective,
                random_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if local_result3.fun < best_score:
                best_score = local_result3.fun
                best_pts = local_result3.x.reshape(N, D)
        except Exception:
            pass
    
    # Final polish
    if best_pts is not None and _time_remaining() > 5:
        try:
            final_result = minimize(
                _objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if final_result.fun < best_score:
                best_score = final_result.fun
                best_pts = final_result.x.reshape(N, D)
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return Fibonacci sphere
        return init_pts
    
    return best_pts
# EVOLVE-BLOCK-END
