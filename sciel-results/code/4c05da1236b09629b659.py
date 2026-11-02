import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Pre-compute all triangle index combinations
    tri_indices = list(combinations(range(N), 3))
    tri_indices = np.array(tri_indices)  # shape (364, 3)
    
    def compute_score(pts):
        """Compute min triangle area / convex hull area"""
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
        
        pi = pts[tri_indices[:, 0]]
        pj = pts[tri_indices[:, 1]]
        pk = pts[tri_indices[:, 2]]
        
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / hull_area
    
    def neg_score(flat_pts):
        pts = flat_pts.reshape(N, 2)
        if np.allclose(pts, pts[0]):
            return 0.0
        return -compute_score(pts)
    
    best_score_val = 0
    best_result = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    initial_configs.append(x0)
    
    # 2. Perturbed regular polygon with different seeds
    rng = np.random.RandomState(42)
    for scale in [0.05, 0.1, 0.2, 0.3]:
        x_pert = x0 + rng.normal(0, scale, x0.shape)
        initial_configs.append(x_pert)
    
    # 3. Two concentric heptagons (inner and outer)
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_angles = outer_angles + np.pi/7
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
    x_conc = np.vstack([outer_pts, inner_pts]).flatten()
    initial_configs.append(x_conc)
    
    # 4. Random points in unit circle
    for seed in [1, 2, 3]:
        rng2 = np.random.RandomState(seed)
        r = rng2.uniform(0.3, 1.0, N)
        theta = rng2.uniform(0, 2*np.pi, N)
        x_rand = np.column_stack([r*np.cos(theta), r*np.sin(theta)]).flatten()
        initial_configs.append(x_rand)
    
    # Optimize each configuration
    for x_start in initial_configs:
        if time.time() > deadline - 15:
            break
        
        # Try Powell first
        try:
            res = minimize(neg_score, x_start, method='Powell',
                          options={'maxiter': 300, 'ftol': 1e-12, 'disp': False})
            score_val = -res.fun
            if score_val > best_score_val:
                best_score_val = score_val
                best_result = res.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 10:
            break
        
        # Try Nelder-Mead as backup
        try:
            res_nm = minimize(neg_score, x_start, method='Nelder-Mead',
                            options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            score_val = -res_nm.fun
            if score_val > best_score_val:
                best_score_val = score_val
                best_result = res_nm.x.copy()
        except Exception:
            pass
    
    # Final polish on best result
    if time.time() < deadline - 5 and best_result is not None:
        try:
            res_polish = minimize(neg_score, best_result, method='Powell',
                                  options={'maxiter': 150, 'ftol': 1e-14, 'disp': False})
            score_val = -res_polish.fun
            if score_val > best_score_val:
                best_score_val = score_val
                best_result = res_polish.x.copy()
        except Exception:
            pass
    
    # If no optimization succeeded, return regular polygon
    if best_result is None:
        best_result = x0
    
    return best_result.reshape(N, 2)
