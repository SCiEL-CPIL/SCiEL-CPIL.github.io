"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_start_time = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _min_triangle_area(pts):
    """Compute min area over all triples."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pj = pts[j]
            for k in range(j + 1, n):
                pk = pts[k]
                cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                a = 0.5 * abs(cross)
                if a < min_a:
                    min_a = a
    return min_a


def _convex_hull_area(pts):
    """Compute convex hull area using Andrew's monotone chain."""
    pts = np.array(pts, dtype=float)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    n = len(pts)
    if n < 3:
        return 0.0
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    
    area = 0.0
    m = len(hull)
    for i in range(m):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % m]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-15:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_area


def _soft_min_triangle_area(pts, beta=50.0):
    """Smooth surrogate for min triangle area using log-sum-exp."""
    n = len(pts)
    areas = []
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pj = pts[j]
            for k in range(j + 1, n):
                pk = pts[k]
                cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                a = 0.5 * abs(cross)
                areas.append(a)
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    max_a = np.max(areas)
    log_sum = max_a + np.log(np.sum(np.exp(-beta * (areas - max_a))))
    return log_sum / beta  # This is actually a soft-max, we want soft-min
    # Actually for soft-min: -logsumexp(-beta * areas) / beta
    # Let me redo:
    # soft_min = -1/beta * log(sum(exp(-beta * a)))
    # = -1/beta * log(sum(exp(-beta * (a - min_a)))) - min_a
    # Let's use a stable version


def _soft_min_triangle_area_stable(pts, beta=30.0):
    """Stable soft-min of triangle areas."""
    n = len(pts)
    areas = np.array([
        0.5 * abs((pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) - 
                  (pts[j][1] - pts[i][1]) * (pts[k][0] - pts[i][0]))
        for i in range(n) for j in range(i+1, n) for k in range(j+1, n)
    ])
    if len(areas) == 0:
        return 0.0
    min_a = np.min(areas)
    # soft_min ≈ min_a + (1/beta) * log(sum(exp(-beta*(a - min_a))))
    # But this is not quite right. Let's use:
    # f(a) = -1/beta * log(sum(exp(-beta * a)))
    # To make it stable:
    # = -1/beta * log(sum(exp(-beta * (a - max_a)))) - max_a
    # Hmm, let me just use a simple approach
    max_val = np.max(areas)
    shifted = areas - max_val
    log_sum = max_val + np.log(np.sum(np.exp(-beta * shifted)))
    # soft_min = -log_sum / beta ... no
    # Actually: soft_min(a) = -1/beta * log(sum(exp(-beta * a_i)))
    # = -1/beta * log(sum(exp(-beta * (a_i - max_a)) * exp(-beta * max_a)))
    # = -1/beta * [log(sum(exp(-beta * (a_i - max_a)))) + (-beta * max_a)]
    # = -1/beta * log(sum(exp(-beta * (a_i - max_a)))) + max_a
    # Wait, that gives soft_max not soft_min.
    
    # For soft_min: we want to approximate min(a_i)
    # soft_min = -1/beta * log(sum(exp(-beta * a_i)))
    # Stable: let m = max(a_i), then
    # = -1/beta * log(sum(exp(-beta * (a_i - m)))) - m
    # No: sum(exp(-beta * a_i)) = exp(-beta * m) * sum(exp(-beta * (a_i - m)))
    # log(sum(exp(-beta * a_i))) = -beta * m + log(sum(exp(-beta * (a_i - m))))
    # soft_min = -1/beta * (-beta * m + log(sum(exp(-beta * (a_i - m)))))
    # = m - 1/beta * log(sum(exp(-beta * (a_i - m))))
    # But a_i - m <= 0, so -beta * (a_i - m) >= 0, exp of that is >= 1
    # This should be stable.
    exp_terms = np.exp(-beta * (areas - max_val))
    log_sum_exp = np.log(np.sum(exp_terms))
    soft_min_val = max_val - (1.0 / beta) * log_sum_exp
    return soft_min_val


def _objective_4d(params):
    """Objective for 4D symmetric two-ring parameterization.
    params: [r_hull, r_inner, rot_hull, rot_inner]
    Returns negative score (for minimization).
    """
    r_hull, r_inner, rot_hull, rot_inner = params
    if r_hull <= 0 or r_inner < 0:
        return 1e6
    
    n_hull = 7  # 7 points on outer ring
    n_inner = 6  # 6 points on inner ring
    
    points = np.zeros((N, 2))
    
    # Outer ring: regular 7-gon
    angles_hull = np.linspace(0, 2 * np.pi, n_hull, endpoint=False) + rot_hull
    points[:n_hull, 0] = r_hull * np.cos(angles_hull)
    points[:n_hull, 1] = r_hull * np.sin(angles_hull)
    
    # Inner ring: regular 6-gon
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rot_inner
    points[n_hull:, 0] = r_inner * np.cos(angles_inner)
    points[n_hull:, 1] = r_inner * np.sin(angles_inner)
    
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    min_ta = _min_triangle_area(points)
    score = min_ta / hull_area
    return -score


def _objective_full(pts_flat):
    """Objective for full 26D space using soft-min surrogate."""
    pts = pts_flat.reshape(N, 2)
    
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 1e6
    
    # Use soft-min for smoothness
    beta = 20.0
    soft_min_ta = _soft_min_triangle_area_stable(pts, beta)
    
    score = soft_min_ta / hull_area
    return -score


def construct():
    best_score = 0.0
    best_points = None
    
    # Initial: regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular = np.column_stack((np.cos(angles), np.sin(angles)))
    s_reg = _score(regular)
    best_score = s_reg
    best_points = regular.copy()
    
    if _elapsed() > TIME_LIMIT - 5:
        return best_points
    
    # Try 4D symmetric optimization from multiple starts
    starts_4d = [
        [1.0, 0.5, 0.0, 0.0],
        [1.0, 0.4, 0.0, np.pi/6],
        [1.0, 0.6, 0.0, np.pi/7],
        [1.0, 0.3, 0.0, np.pi/12],
        [1.0, 0.5, np.pi/14, 0.0],
        [1.0, 0.45, 0.1, 0.2],
        [1.0, 0.55, -0.1, -0.15],
    ]
    
    for start in starts_4d:
        if _elapsed() > TIME_LIMIT - 10:
            break
        try:
            res = minimize(_objective_4d, start, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success or res.fun < 0:
                r_hull, r_inner, rot_hull, rot_inner = res.x
                n_hull = 7
                n_inner = 6
                pts = np.zeros((N, 2))
                angles_hull = np.linspace(0, 2 * np.pi, n_hull, endpoint=False) + rot_hull
                pts[:n_hull, 0] = r_hull * np.cos(angles_hull)
                pts[:n_hull, 1] = r_hull * np.sin(angles_hull)
                angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rot_inner
                pts[n_hull:, 0] = r_inner * np.cos(angles_inner)
                pts[n_hull:, 1] = r_inner * np.sin(angles_inner)
                
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass
    
    if _elapsed() > TIME_LIMIT - 10:
        return best_points
    
    # Full-space optimization from best point found
    try:
        x0 = best_points.flatten()
        res = minimize(_objective_full, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.success or res.fun < 0:
            pts = res.x.reshape(N, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
    except Exception:
        pass
    
    if _elapsed() > TIME_LIMIT - 5:
        return best_points
    
    # Final polish: try a few more Nelder-Mead refinements
    for _ in range(3):
        if _elapsed() > TIME_LIMIT - 3:
            break
        try:
            x0 = best_points.flatten()
            res = minimize(_objective_full, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.success or res.fun < 0:
                pts = res.x.reshape(N, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
