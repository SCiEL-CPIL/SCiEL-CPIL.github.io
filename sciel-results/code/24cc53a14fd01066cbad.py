import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 1.0

def _score(points):
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(params, beta=100.0):
    """Negative smooth min area / hull area.
    We minimize - (soft_min(areas) / hull_area).
    """
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    
    # Soft min: -logsumexp(-beta * areas) / beta
    # To avoid overflow in exp, we subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    # If max_val is very large negative, exp will underflow to 0, sum will be 0, log(0) is -inf.
    # But areas are positive, so neg_beta_areas are negative.
    lse = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -lse / beta
    
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e10
    
    return -soft_min / hull_area

def construct():
    n = N
    t0 = time.time()
    deadline = 130.0 # Keep 20s safety margin
    
    # Start with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points_init = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = points_init.copy()
    best_score = _score(best_points)
    
    # Optimization 1: L-BFGS-B with numerical gradients
    # This is generally faster and more robust than Nelder-Mead for smooth surrogates
    try:
        x0 = points_init.flatten()
        # Use a moderate beta for L-BFGS-B to ensure smoothness
        res1 = minimize(_soft_min_objective, x0, args=(50.0,), method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9})
        
        if time.time() - t0 > deadline:
            return best_points
            
        candidate = res1.x.reshape(-1, 2)
        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_points = candidate.copy()
    except Exception:
        pass
    
    # Optimization 2: Powell method (derivative-free, often good for such problems)
    # Start from the best point found so far
    if time.time() - t0 < deadline:
        try:
            x0 = best_points.flatten()
            # Powell can be slow, limit iterations
            res2 = minimize(_soft_min_objective, x0, args=(100.0,), method='Powell',
                            options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12})
            
            if time.time() - t0 > deadline:
                return best_points
                
            candidate = res2.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
            
    # Optimization 3: Nelder-Mead Polish
    # Final high-precision polish
    if time.time() - t0 < deadline:
        try:
            x0 = best_points.flatten()
            # Use a very high beta to make the objective very close to the true min
            res3 = minimize(_soft_min_objective, x0, args=(200.0,), method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() - t0 > deadline:
                return best_points
                
            candidate = res3.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
            
    # Final check: ensure we return the best found
    return best_points
