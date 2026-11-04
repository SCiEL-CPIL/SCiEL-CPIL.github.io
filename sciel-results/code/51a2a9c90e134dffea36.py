from scipy.optimize import minimize
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 14

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    n = len(points)
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    p_i = points[i_idx[mask]]
    p_j = points[j_idx[mask]]
    p_k = points[k_idx[mask]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area


def _superellipse_points(n_exp, aspect, n_pts=N):
    """
    Generate n_pts points uniformly distributed by arc length on a superellipse.
    Superellipse: |x/a|^n_exp + |y/b|^n_exp = 1
    where a = 1.0 (fixed scale) and b = aspect * a.
    Points are placed at equal arc-length intervals along the curve.
    """
    a = 1.0
    b = aspect * a
    
    # Parametrize superellipse: x = a * sign(cos(t)) * |cos(t)|^(2/n_exp), y = b * sign(sin(t)) * |sin(t)|^(2/n_exp)
    # This gives a closed curve. We sample t uniformly, then reparameterize by arc length.
    
    # First, sample many points to estimate arc length
    n_sample = 1000
    t_vals = np.linspace(0, 2 * np.pi, n_sample, endpoint=False)
    
    cos_t = np.cos(t_vals)
    sin_t = np.sin(t_vals)
    
    x = a * np.sign(cos_t) * np.abs(cos_t) ** (2.0 / n_exp)
    y = b * np.sign(sin_t) * np.abs(sin_t) ** (2.0 / n_exp)
    
    pts = np.column_stack([x, y])
    
    # Compute cumulative arc length
    diffs = np.diff(pts, axis=0, append=pts[:1])
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_length = np.cumsum(seg_lengths)
    total_length = cum_length[-1]
    
    # We want n_pts points at equal arc-length intervals
    target_lengths = np.linspace(0, total_length, n_pts, endpoint=False)
    
    # Interpolate to get points at target arc lengths
    # cum_length is monotonically increasing, so we can use searchsorted
    indices = np.searchsorted(cum_length, target_lengths)
    indices = np.clip(indices, 0, n_sample - 1)
    
    # For better accuracy, interpolate within each segment
    # For simplicity, just take the closest sample point
    result = pts[indices]
    
    return result


def _score_superellipse(params):
    """
    Score function for superellipse parameters: [n_exp, aspect]
    """
    n_exp, aspect = params
    if n_exp <= 0 or aspect <= 0:
        return 0.0
    
    points = _superellipse_points(n_exp, aspect)
    return _score(points)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_score = 0
    best_points = None
    
    # Fallback: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg14 = np.column_stack([np.cos(angles), np.sin(angles)])
    reg14_score = _score(reg14)
    best_score = reg14_score
    best_points = reg14
    
    # Strategy: Superellipse with shape exponent n_exp and aspect ratio
    # n_exp=2 is a circle/ellipse, n_exp=1 is a diamond (L1), n_exp>2 approaches a square
    # aspect = b/a, where a=1 fixed
    
    # Grid search over n_exp and aspect
    n_exp_vals = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
    aspect_vals = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0])
    
    for n_exp in n_exp_vals:
        if time.time() > deadline - 5:
            break
        for aspect in aspect_vals:
            if time.time() > deadline - 5:
                break
            s = _score_superellipse([n_exp, aspect])
            if s > best_score:
                best_score = s
                best_points = _superellipse_points(n_exp, aspect)
    
    # Local optimization from best grid point
    if best_score > 0 and time.time() < deadline - 20:
        # Find which grid point gave the best score
        best_n_exp = None
        best_aspect = None
        for n_exp in n_exp_vals:
            for aspect in aspect_vals:
                s = _score_superellipse([n_exp, aspect])
                if s > best_score - 1e-10:
                    best_n_exp = n_exp
                    best_aspect = aspect
                    break
            if best_n_exp is not None:
                break
        
        if best_n_exp is None:
            best_n_exp = 2.0
            best_aspect = 1.0
        
        def _neg_score_se(params):
            return -_score_superellipse(params)
        
        # Try Nelder-Mead from the best grid point
        try:
            res = minimize(_neg_score_se, [best_n_exp, best_aspect], method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True,
                                    'maxfev': 5000})
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = _superellipse_points(res.x[0], res.x[1])
        except Exception:
            pass
    
    # Additional restarts from different initial points
    if time.time() < deadline - 20:
        init_points = [
            [1.5, 1.0],
            [2.0, 0.8],
            [3.0, 1.2],
            [4.0, 0.9],
            [2.5, 1.1],
        ]
        
        for init in init_points:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(_neg_score_se, init, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True,
                                        'maxfev': 3000})
                if res.fun < -best_score:
                    best_score = -res.fun
                    best_points = _superellipse_points(res.x[0], res.x[1])
            except Exception:
                pass
    
    # Final validation
    if best_points is None:
        return reg14
    
    final_score = _score(best_points)
    if final_score <= 0:
        return reg14
    
    return best_points
