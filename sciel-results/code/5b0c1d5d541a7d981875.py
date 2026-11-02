import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit
    
    # Pre-compute all triangle index combinations
    tri_indices = list(combinations(range(N), 3))
    tri_indices = np.array(tri_indices)  # shape (364, 3)
    
    def compute_score(pts):
        """Compute min triangle area / convex hull area"""
        # Check for degenerate points
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            return 0.0
            
        # Convex Hull Area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume  # For 2D, .volume is area
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
        
        # Vectorized min triangle area computation
        pi = pts[tri_indices[:, 0]]  # shape (364, 2)
        pj = pts[tri_indices[:, 1]]  # shape (364, 2)
        pk = pts[tri_indices[:, 2]]  # shape (364, 2)
        
        v1 = pj - pi  # shape (364, 2)
        v2 = pk - pi  # shape (364, 2)
        
        # Cross product: v1[0]*v2[1] - v1[1]*v2[0]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        if min_area < 1e-10:
            return 0.0
            
        score = min_area / hull_area
        return score
    
    def neg_score(flat_pts):
        pts = flat_pts.reshape(N, 2)
        return -compute_score(pts)
    
    # Bounds for Differential Evolution
    # Since the score is scale-invariant, we can constrain to a box.
    # If points are too close, hull area -> 0, score -> 0.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_score_val = 0
    best_result = None
    
    # 1. Differential Evolution for global search
    # DE is stochastic, so we set a seed for reproducibility if needed, 
    # but random is fine.
    try:
        # popsize=20, maxiter=50 might be too slow? 
        # 14 points * 2 coords = 28 dimensions.
        # DE with popsize 20 is 20*28 = 560 evals per generation.
        # 50 generations = 28000 evals.
        # Each eval is fast (vectorized numpy + ConvexHull).
        # ConvexHull for 14 points is ~microseconds.
        # Let's try a moderate budget.
        
        de_result = differential_evolution(
            neg_score, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-5, 
            seed=42,
            polish=False, # We will do our own polish
            updating='immediate',
            disp=False
        )
        
        if time.time() < deadline - 10:
            score_val = -de_result.fun
            if score_val > best_score_val:
                best_score_val = score_val
                best_result = de_result.x.copy()
                
    except Exception as e:
        pass
    
    # 2. Local Refinement
    if best_result is None:
        # Fallback to regular polygon if DE failed
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_result = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # Polish with Powell
    if time.time() < deadline - 5:
        try:
            res_polish = minimize(
                neg_score, 
                best_result, 
                method='Powell',
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            score_val = -res_polish.fun
            if score_val > best_score_val:
                best_score_val = score_val
                best_result = res_polish.x.copy()
        except Exception:
            pass
            
    # 3. Final Polish with Nelder-Mead (sometimes better for non-smooth min)
    if time.time() < deadline - 2:
        try:
            res_nm = minimize(
                neg_score, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
            )
            score_val = -res_nm.fun
            if score_val > best_score_val:
                best_score_val = score_val
                best_result = res_nm.x.copy()
        except Exception:
            pass

    return best_result.reshape(N, 2)
