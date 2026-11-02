import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Pre-compute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_indices = np.array(tri_indices)
    I, J, K = tri_indices[:, 0], tri_indices[:, 1], tri_indices[:, 2]
    
    def compute_score(pts):
        # pts is (N, 2)
        if np.allclose(pts, pts[0]):
            return 0.0
            
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        Pi = pts[I]
        Pj = pts[J]
        Pk = pts[K]
        
        v1 = Pj - Pi
        v2 = Pk - Pi
        
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        return min_area / hull_area

    def neg_score_centered(flat_pts):
        pts = flat_pts.reshape(N, 2)
        # Center the points to remove translation invariance
        pts = pts - np.mean(pts, axis=0)
        return -compute_score(pts)

    # Initial guess: Regular 14-gon, centered
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    x0 = x0 - np.mean(x0, axis=0) # Should be 0
    x0_flat = x0.flatten()
    
    best_result = x0_flat
    best_score_val = compute_score(x0)
    
    # Optimization 1: Powell
    try:
        res = minimize(neg_score_centered, x0_flat, method='Powell', 
                       options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
        if -res.fun > best_score_val:
            best_score_val = -res.fun
            best_result = res.x
    except Exception:
        pass

    # Optimization 2: Nelder-Mead
    if time.time() < deadline - 10:
        x0_pert = x0_flat + np.random.normal(0, 0.1, x0_flat.shape)
        try:
            res2 = minimize(neg_score_centered, x0_pert, method='Nelder-Mead', 
                            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
            if -res2.fun > best_score_val:
                best_score_val = -res2.fun
                best_result = res2.x
        except Exception:
            pass

    # Optimization 3: Final Polish with Powell
    if time.time() < deadline - 5:
        try:
            res3 = minimize(neg_score_centered, best_result, method='Powell', 
                            options={'maxiter': 2000, 'ftol': 1e-16, 'disp': False})
            if -res3.fun > best_score_val:
                best_score_val = -res3.fun
                best_result = res3.x
        except Exception:
            pass

    # Return the best result found
    return best_result.reshape(N, 2)
