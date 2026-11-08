"""Heilbronn convex regions n=14: maximize minimum triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

N = 14

# EVOLVE-BLOCK-START
def _triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points.
    Returns array of shape (n*(n-1)*(n-2)/6,)."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    return np.array(areas)

def _convex_hull_area(pts):
    """Compute the area of the convex hull of the point set."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if points are degenerate, return a small value
        return 1e-10

def _score(pts):
    """Score = min triangle area / convex hull area."""
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _neg_smooth_min_triangle_area(pts_flat, temperature=0.05):
    """Negative smooth approximation of the minimum triangle area using log-sum-exp.
    We maximize the smooth min, so minimize the negative.
    
    The smooth min is: -1/T * log(sum(exp(-T * area_i)))
    But we want to maximize min(area_i), so we minimize -min(area_i).
    Using log-sum-exp: smooth_min = -1/T * log(sum(exp(-T * a_i)))
    This approximates min(a_i) as T -> 0.
    
    Actually, let's use: f = -smooth_min = 1/T * log(sum(exp(-T * a_i)))
    Wait, that's not right either. Let me think again.
    
    min(a_i) = -1/T * log(sum(exp(-T * a_i))) as T -> 0+
    So to maximize min(a_i), we minimize -min(a_i) = 1/T * log(sum(exp(-T * a_i)))
    
    But this has a numerical issue. Let's use a different approach:
    We want to maximize the minimum area. Let's directly work with the minimum.
    For L-BFGS-B, we need a smooth objective. The log-sum-exp trick:
    
    smooth_min(a, T) = -1/T * log(sum(exp(-T * a_i)))
    
    This is a smooth lower bound on min(a_i) that approaches it as T -> 0.
    To maximize smooth_min, we minimize -smooth_min = 1/T * log(sum(exp(-T * a_i)))
    
    However, for very small T, this becomes numerically unstable. Let's use a moderate T.
    """
    pts = pts_flat.reshape(-1, 2)
    areas = _triangle_areas(pts)
    T = temperature
    # Numerically stable log-sum-exp
    min_a = np.min(areas)
    shifted = -T * (areas - min_a)
    log_sum_exp = min_a * T + np.log(np.sum(np.exp(-shifted)))
    # smooth_min = -1/T * log_sum_exp_of_negated
    # Actually: smooth_min = -1/T * log(sum(exp(-T * a_i)))
    # Let's compute it properly:
    # sum(exp(-T * a_i)) = exp(-T * min_a) * sum(exp(-T * (a_i - min_a)))
    # log(sum(exp(-T * a_i))) = -T * min_a + log(sum(exp(-T * (a_i - min_a))))
    # smooth_min = -1/T * [-T * min_a + log(sum(exp(-T * (a_i - min_a))))]
    #             = min_a - 1/T * log(sum(exp(-T * (a_i - min_a))))
    # The term in log is <= log(n_triangles) since exp(-T*(a_i-min_a)) <= 1
    # So smooth_min <= min_a, which is correct (it's a lower bound).
    
    # To maximize smooth_min, minimize -smooth_min
    neg_smooth_min = -smooth_min
    return neg_smooth_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    best_pts = None
    best_score = 0.0
    
    # Start with a regular polygon as initial guess
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_pts = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Try multiple random initializations and the regular polygon
    rng = np.random.default_rng(N)
    
    initializations = [init_pts]
    for _ in range(20):
        # Random points in a unit disk
        r = np.sqrt(rng.random(N))
        angles = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        initializations.append(pts)
    
    for init in initializations:
        if time.time() > deadline:
            break
        
        # Evaluate the initial point
        init_score = _score(init)
        if init_score > best_score:
            best_score = init_score
            best_pts = init.copy()
        
        # Use L-BFGS-B to optimize the smooth objective
        # We optimize the negative smooth min triangle area, but we also need to
        # account for the hull area normalization. Since the hull area changes,
        # we can't easily separate them. Let's just maximize the raw min area
        # and then normalize, or we can try to optimize the ratio directly.
        
        # For now, let's try maximizing the raw minimum triangle area (smoothed)
        # and see if that helps. The hull area for points in a bounded region
        # won't vary too wildly.
        
        x0 = init.flatten()
        
        def objective(x):
            return _neg_smooth_min_triangle_area(x, temperature=0.1)
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_pts = result.x.reshape(-1, 2)
            opt_score = _score(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
                
        except Exception as e:
            pass
    
    # Additional refinement: try to directly optimize the ratio using a
    # different smooth approximation
    if best_pts is not None and time.time() < deadline:
        # Try a few more local optimizations with different temperatures
        for T in [0.05, 0.2, 0.5]:
            if time.time() > deadline:
                break
            
            x0 = best_pts.flatten()
            
            def objective_ratio(x, T=T):
                pts = x.reshape(-1, 2)
                areas = _triangle_areas(pts)
                hull_area = _convex_hull_area(pts)
                if hull_area < 1e-10:
                    return 0.0
                # Smooth min of areas / hull_area
                # We want to maximize this, so minimize the negative
                min_a = np.min(areas)
                shifted = -T * (areas - min_a)
                log_sum = min_a * T + np.log(np.sum(np.exp(-shifted)))
                smooth_min = -1.0 / T * log_sum
                return -smooth_min / hull_area
            
            try:
                result = minimize(
                    objective_ratio,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                opt_pts = result.x.reshape(-1, 2)
                opt_score = _score(opt_pts)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts.copy()
                    
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback: return the regular polygon
        best_pts = init_pts
    
    return best_pts
# EVOLVE-BLOCK-END
