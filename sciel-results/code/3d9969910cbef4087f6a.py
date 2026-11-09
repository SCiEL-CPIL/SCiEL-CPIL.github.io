# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _compute_score(points):
    """Compute the minimum vertex score for a given polygon."""
    n = len(points)
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    dists_to_centroid = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    pts = pts / mean_dist
    
    # Check strict convexity
    edges = np.roll(pts, -1, axis=0) - pts
    cross = np.array([edges[i][0]*edges[(i+1)%n][1] - edges[i][1]*edges[(i+1)%n][0] for i in range(n)])
    if np.any(np.abs(cross) < 1e-12):
        return 0.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                return 0.0
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts, axis=1)
        d_sorted = np.sort(np.delete(d, i))
        if np.min(d_sorted) < 0.05:
            min_score = 0.0
            break
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            min_score = 0.0
            break
        if k == 0:
            four = d_sorted[0:4]
        elif k == 17:
            four = d_sorted[15:19]
        else:
            four = d_sorted[k-1:k+3]
        ratios = np.maximum(four / D, D / four)
        raw = np.mean(ratios)
        score = min(1.0, 1.0 / raw)
        min_score = min(min_score, score)
    return min_score

def _neg_score(points):
    return -_compute_score(points)

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    # Regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x0 = np.cos(angles)
    y0 = np.sin(angles)
    best_points = np.stack([x0, y0], axis=1)
    best_score = _compute_score(best_points)
    
    # Use differential evolution for global search
    bounds = [(-2, 2)] * (2*n)
    
    np.random.seed(42)
    
    # Run differential evolution
    try:
        if time.time() < deadline - 60:
            result = differential_evolution(
                _neg_score, bounds, 
                seed=42, maxiter=200, tol=1e-6,
                polish=True, init='sobol',
                workers=1
            )
            if result.fun < -best_score:
                pts = result.x.reshape(n, 2)
                s = _compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
    except Exception:
        pass
    
    # Local refinement with L-BFGS-B from best point
    if time.time() < deadline - 30:
        try:
            init = best_points.flatten()
            result = minimize(_neg_score, init, method='L-BFGS-B',
                             options={'maxiter': 1000, 'maxfun': 5000})
            pts = result.x.reshape(n, 2)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass
    
    # Multiple random restarts with local optimization
    np.random.seed(123)
    for restart in range(20):
        if time.time() > deadline - 15:
            break
        
        # Random convex-ish polygon: perturb regular polygon
        noise = np.random.randn(2*n) * 0.3
        init = np.concatenate([x0, y0]) + noise
        
        try:
            result = minimize(_neg_score, init, method='L-BFGS-B',
                             options={'maxiter': 300, 'maxfun': 1000})
            pts = result.x.reshape(n, 2)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-END
