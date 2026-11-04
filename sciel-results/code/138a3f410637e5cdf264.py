"""Heilbronn convex regions n=14: maximize minimum triangle area / hull area."""
N = 14
import numpy as np
import time

_DEADLINE = time.time() + 140  # safety margin before 150s kill

def _min_triangle_area(pts):
    """Compute minimum triangle area for all triples."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                ax = pts[j][0] - pts[i][0]
                ay = pts[j][1] - pts[i][1]
                bx = pts[k][0] - pts[i][0]
                by = pts[k][1] - pts[i][1]
                area = 0.5 * abs(ax * by - ay * bx)
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area using scipy."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace on all points sorted by angle
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        order = np.argsort(angles)
        p = pts[order]
        n = len(p)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += p[i][0] * p[j][1] - p[j][0] * p[i][1]
        return abs(area) / 2.0

def _score(pts):
    """Score = min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    mta = _min_triangle_area(pts)
    return mta / ha

def construct():
    """Place 14 points to maximize minimum triangle area / hull area."""
    t_start = time.time()
    
    # Start with equally spaced points on a unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = _score(best_pts)
    
    # Try multiple initial configurations and local optimization
    from scipy.optimize import minimize
    
    def neg_score(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    # Different initial configurations
    configs = [
        # Equally spaced on circle
        np.column_stack([np.cos(angles), np.sin(angles)]),
        # Two concentric rings
        np.vstack([
            np.column_stack([np.cos(angles[:7]), np.sin(angles[:7])]),
            0.5 * np.column_stack([np.cos(angles[7:] + 0.1), np.sin(angles[7:] + 0.1)])
        ]),
        # Regular 14-gon scaled
        np.column_stack([np.cos(angles) * 0.8, np.sin(angles) * 0.8]),
    ]
    
    for init in configs:
        if time.time() > _DEADLINE:
            break
        
        # Try Nelder-Mead optimization
        try:
            res = minimize(neg_score, init.flatten(), method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_pts = res.x.reshape(-1, 2)
        except Exception:
            pass
        
        # Try L-BFGS-B
        if time.time() > _DEADLINE:
            break
        try:
            res = minimize(neg_score, init.flatten(), method='L-BFGS-B',
                          options={'maxiter': 100, 'gtol': 1e-6})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_pts = res.x.reshape(-1, 2)
        except Exception:
            pass
    
    # Normalize: center at origin, scale so hull area = 1
    cx = best_pts[:, 0].mean()
    cy = best_pts[:, 1].mean()
    best_pts -= np.array([cx, cy])
    
    ha = _hull_area(best_pts)
    if ha > 1e-12:
        scale = np.sqrt(1.0 / ha)
        best_pts *= scale
    
    return best_pts
