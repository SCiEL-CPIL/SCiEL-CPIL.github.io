"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _score(points):
    """Compute the score: Delta / n^n where Delta is product of ordered-pair distances
    after scaling to diameter 2."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    # Compute all pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Diameter is the max pairwise distance
    diameter = np.max(dists)
    if diameter < 1e-15:
        return 0.0
    # Scale to diameter 2
    scaled = dists / diameter * 2.0
    # Product over ordered pairs (i != j)
    # Use log to avoid overflow
    mask = ~np.eye(n, dtype=bool)
    log_delta = np.sum(np.log(scaled[mask] + 1e-300))
    score = math.exp(log_delta / (n ** n))
    return score

def _log_score(points):
    """Compute log of the score for optimization."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    diameter = np.max(dists)
    if diameter < 1e-15:
        return -np.inf
    scaled = dists / diameter * 2.0
    mask = ~np.eye(n, dtype=bool)
    log_delta = np.sum(np.log(scaled[mask] + 1e-300))
    return log_delta - n ** n * 0  # We want to maximize log_delta, score = exp(log_delta/n^n)
    # Actually score = exp(log_delta / n^n), so maximizing score = maximizing log_delta

def _log_delta(pts):
    """Log of product of ordered-pair distances after scaling to diameter 2."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    diameter = np.max(dists)
    if diameter < 1e-15:
        return -np.inf
    scaled = dists / diameter * 2.0
    mask = ~np.eye(n, dtype=bool)
    return np.sum(np.log(scaled[mask] + 1e-300))

def _regular_octagon():
    from math import cos, sin, pi
    return np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])

def _optimize_from_start(start_pts, max_iters=2000, tol=1e-10):
    """Gradient-free optimization using Nelder-Mead style approach on 16 coordinates.
    We optimize log_delta (product of distances / diameter^56)."""
    from scipy.optimize import minimize
    
    n = 8
    x0 = start_pts.flatten()
    
    def objective(x):
        pts = x.reshape(n, 2)
        return -_log_delta(pts)
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iters, 'xatol': tol, 'fatol': tol, 'adaptive': True})
    return result.x.reshape(n, 2)

def _refine(pts, max_iters=5000, tol=1e-12):
    """High-precision refinement using BFGS-like approach with careful handling."""
    from scipy.optimize import minimize
    
    n = 8
    x0 = pts.flatten()
    
    def objective(x):
        pts = x.reshape(n, 2)
        return -_log_delta(pts)
    
    def gradient(x):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        base = -_log_delta(x.reshape(n, 2))
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            grad[i] = (-_log_delta(xp.reshape(n, 2)) - base) / eps
        return grad
    
    result = minimize(objective, x0, jac=gradient, method='BFGS',
                      options={'maxiter': max_iters, 'gtol': tol, 'disp': False})
    return result.x.reshape(n, 2)

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    best_pts = _regular_octagon()
    best_score = _score(best_pts)
    
    # Try multiple random initializations and optimize
    rng = np.random.default_rng(42)
    
    # Start with regular octagon and perturbations
    starts = [best_pts.copy()]
    
    # Add several perturbed versions
    for _ in range(15):
        angle_offset = rng.uniform(0, 2 * math.pi)
        radii = 1.0 + rng.normal(0, 0.1, 8)
        angles = 2 * math.pi * np.arange(8) / 8 + angle_offset + rng.normal(0, 0.05, 8)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        starts.append(pts)
    
    # Also try some structured non-regular configurations
    # Two concentric squares
    for r1, r2, rot in [(0.5, 1.0, 0.0), (0.7, 1.0, math.pi/8), (0.3, 1.0, math.pi/8)]:
        angles1 = 2 * math.pi * np.arange(4) / 4
        angles2 = 2 * math.pi * np.arange(4) / 4 + rot
        pts = np.vstack([
            np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)]),
            np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        ])
        starts.append(pts)
    
    # Optimize each start
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        try:
            # First pass: Nelder-Mead
            opt_pts = _optimize_from_start(start, max_iters=1000, tol=1e-8)
            score = _score(opt_pts)
            if score > best_score:
                best_score = score
                best_pts = opt_pts
            
            # Second pass: high-precision refinement
            if time.time() > deadline - 20:
                break
            refined = _refine(opt_pts, max_iters=3000, tol=1e-12)
            score2 = _score(refined)
            if score2 > best_score:
                best_score = score2
                best_pts = refined
        except Exception:
            continue
    
    # Final polish on the best
    if time.time() < deadline - 5:
        try:
            final = _refine(best_pts, max_iters=5000, tol=1e-14)
            final_score = _score(final)
            if final_score > best_score:
                best_pts = final
                best_score = final_score
        except Exception:
            pass
    
    # Return as list of [x, y]
    return [[float(p[0]), float(p[1])] for p in best_pts]
# EVOLVE-BLOCK-END
