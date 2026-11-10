# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 620.0  # safety margin before 630s kill


def _vertex_scores(pts):
    """Compute per-vertex equidistance scores for a polygon."""
    n = len(pts)
    scores = np.zeros(n)
    for i in range(n):
        d = np.sort(np.linalg.norm(pts - pts[i], axis=1))
        d = d[1:]  # exclude self
        if np.any(d < 0.05):
            scores[i] = 0.0
            continue
        gaps = np.diff(d)
        k = int(np.argmin(gaps))
        D = (d[k] + d[k + 1]) / 2.0
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            sel = d[0:4]
        elif k == 17:
            sel = d[15:19]
        else:
            sel = d[k-1:k+3]
        ratios = np.maximum(sel / D, D / sel)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return scores


def _is_convex(pts):
    """Check strict convexity: all cross products same sign, no repeated points."""
    n = len(pts)
    if n < 3:
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                return False
    # Check all turns same sign
    signs = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        cross = np.cross(p1 - p0, p2 - p1)
        if abs(cross) < 1e-15:
            return False
        signs.append(np.sign(cross))
    if len(set(signs)) != 1:
        return False
    # Total turning should be ±2π (for simple convex polygon, this is automatic)
    return True


def _pts_from_params(params):
    """Convert Fourier parameters to polygon points."""
    n = 20
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones(n)
    # params: [a1, b1, a2, b2, a3, b3, a4, b4, a5, b5]
    for m in range(1, 6):
        a = params[2 * (m - 1)]
        b = params[2 * (m - 1) + 1]
        r += a * np.cos(m * theta) + b * np.sin(m * theta)
    # Ensure r > 0
    r = np.maximum(r, 0.3)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


def _objective(params):
    """Objective: negative of minimum vertex score (to minimize)."""
    pts = _pts_from_params(params)
    if not _is_convex(pts):
        return 10.0  # penalty for non-convex
    scores = _vertex_scores(pts)
    return -np.min(scores)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon maximizing the minimum vertex equidistance score.
    Returns (20, 2) array of vertex coordinates in cyclic order.
    """
    # Start with regular polygon
    n = 20
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    regular = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    
    best_pts = regular.copy()
    best_score = _vertex_scores(best_pts).min()
    
    # Initial params: all zero (regular polygon)
    x0 = np.zeros(10)
    
    # Try multiple random initializations
    np.random.seed(42)
    attempts = []
    
    # Regular polygon as first attempt
    attempts.append(np.zeros(10))
    
    # Small random perturbations
    for _ in range(5):
        attempts.append(np.random.randn(10) * 0.05)
    
    # Slightly larger perturbations
    for _ in range(5):
        attempts.append(np.random.randn(10) * 0.1)
    
    # Some structured perturbations
    for m in range(1, 6):
        p = np.zeros(10)
        p[2*(m-1)] = 0.03
        attempts.append(p)
        p = np.zeros(10)
        p[2*(m-1)+1] = 0.03
        attempts.append(p)
    
    for x0 in attempts:
        if time.time() > _DEADLINE - 60:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            pts = _pts_from_params(result.x)
            if _is_convex(pts):
                scores = _vertex_scores(pts)
                min_score = scores.min()
                if min_score > best_score:
                    best_score = min_score
                    best_pts = pts.copy()
        except Exception:
            continue
    
    # Final check
    if not _is_convex(best_pts):
        # Fallback to regular polygon
        best_pts = regular
    
    return best_pts


# EVOLVE-BLOCK-END
