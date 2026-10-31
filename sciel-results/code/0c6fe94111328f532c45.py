import numpy as np
from scipy.optimize import minimize
import time
import itertools
import scipy.spatial as spat

N = 13
DEADLINE = 140.0 # Safety margin for 150s limit

# Precompute triangle indices for vectorized area calculation
TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))

def _compute_triangle_areas(pts):
    """
    Vectorized computation of areas for all triples of points.
    """
    p1 = pts[TRI_INDICES[:, 0]]
    p2 = pts[TRI_INDICES[:, 1]]
    p3 = pts[TRI_INDICES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area_soft(pts, beta):
    """
    Computes the soft-min of triangle areas using log-sum-exp trick.
    """
    areas = _compute_triangle_areas(pts)
    a_min = np.min(areas)
    # shifted = -beta * (areas - a_min)
    # We want sum(exp(-beta * areas)).
    # Let A_min = min(areas).
    # sum(exp(-beta * A)) = exp(-beta * A_min) * sum(exp(-beta * (A - A_min)))
    # log(sum(...)) = -beta * A_min + log(sum(exp(-beta * (A - A_min))))
    # soft_min = -1/beta * log(sum(...)) = A_min - 1/beta * log(sum(exp(-beta * (A - A_min))))
    
    # To avoid underflow in exp(-beta * (A - A_min)), note that A - A_min >= 0.
    # So -beta * (A - A_min) <= 0.
    # The term closest to 0 exponent is when A = A_min, which is exp(0) = 1.
    # Other terms are exp(negative number), which is fine.
    
    shifted = -beta * (areas - a_min)
    # np.logaddexp.reduce is more stable, but log(sum(exp)) is okay if we subtract min.
    # Since max shifted is 0, exp(shifted) <= 1. No overflow.
    # Sum of exps is at least 1 (from the min term).
    log_sum = np.log(np.sum(np.exp(shifted)))
    return a_min - log_sum / beta

def _score_soft(pts, beta):
    """
    Score using soft-min area and convex hull area.
    """
    try:
        hull = spat.ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    min_area = _min_triangle_area_soft(pts, beta)
    return min_area / hull_area

def _neg_score_soft(params, beta):
    pts = params.reshape(N, 2)
    return -_score_soft(pts, beta)

def _score_hard(pts):
    """
    Exact score calculation for final evaluation.
    """
    try:
        hull = spat.ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

def _generate_initial_configurations():
    """
    Generate several symmetric initial configurations.
    """
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(pts)
    
    # 2. 1 center + 12 ring
    pts = np.zeros((N, 2))
    pts[0] = [0, 0]
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts[1:] = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(pts)
    
    # 3. 1 center + 3 inner + 4 mid + 5 outer
    pts = np.zeros((N, 2))
    idx = 0
    pts[idx] = [0, 0]; idx += 1
    for r, count in [(0.5, 3), (1.0, 4), (1.5, 5)]:
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        offset = 0.5 * 2*np.pi / count if count > 2 else 0
        pts[idx:idx+count] = np.column_stack([r * np.cos(angles + offset), r * np.sin(angles + offset)])
        idx += 1 if count == 1 else count
    configs.append(pts)
    
    # 4. 1 center + 4 inner + 4 mid + 4 outer
    pts = np.zeros((N, 2))
    idx = 0
    pts[idx] = [0, 0]; idx += 1
    for r, count in [(0.5, 4), (1.0, 4), (1.5, 4)]:
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        offset = 0.5 * 2*np.pi / count if count > 2 else 0
        pts[idx:idx+count] = np.column_stack([r * np.cos(angles + offset), r * np.sin(angles + offset)])
        idx += 1 if count == 1 else count
    configs.append(pts)
    
    return configs

def construct():
    start_time = time.time()
    
    configs = _generate_initial_configurations()
    
    best_score = -np.inf
    best_points = None
    
    # We will run a local refinement chain on each initial configuration.
    # The chain uses increasing beta for the soft-min objective to sharpen the landscape.
    
    for init_pts in configs:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        current_points = init_pts.flatten()
        
        # Start with a moderate beta for stability, then increase
        betas = [5.0, 20.0, 100.0, 500.0]
        
        for beta in betas:
            if time.time() - start_time > DEADLINE - 10:
                break
                
            try:
                # L-BFGS-B is good for smooth approximations
                res = minimize(
                    lambda x: _neg_score_soft(x, beta),
                    current_points,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # Update current points to the result for the next iteration
                current_points = res.x
                
                # Evaluate the hard score of the current best candidate
                candidate_pts = res.x.reshape(N, 2)
                hard_score = _score_hard(candidate_pts)
                
                if hard_score > best_score:
                    best_score = hard_score
                    best_points = candidate_pts.copy()
                    
            except Exception:
                continue

    # If we have time left, do a final polish on the best point found
    if best_points is not None and time.time() - start_time < DEADLINE - 5:
        # Use a very high beta for a final smooth refinement
        try:
            res = minimize(
                lambda x: _neg_score_soft(x, 1000.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            candidate_pts = res.x.reshape(N, 2)
            hard_score = _score_hard(candidate_pts)
            if hard_score > best_score:
                best_points = candidate_pts.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = configs[0]
        
    return best_points
