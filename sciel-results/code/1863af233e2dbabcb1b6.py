import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Initial Seed: Regular Polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Precompute indices for triangles
    # i < j < k
    idx = np.array([i for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]).reshape(-1, 3)
    
    def objective(x):
        pts = x.reshape(N, 2)
        # Calculate areas
        p_i = pts[idx[:, 0]]
        p_j = pts[idx[:, 1]]
        p_k = pts[idx[:, 2]]
        
        v1 = p_j - p_i
        v2 = p_k - p_i
        areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        min_area = np.min(areas)
        
        # Calculate Hull Area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 1e10 # Penalize degenerate hulls
            
        if hull_area < 1e-10:
            return 1e10
            
        return -(min_area / hull_area)
        
    best_score = -objective(pts0.flatten())
    best_pts = pts0.copy()
    
    # Run optimization
    # Powell is derivative-free and handles non-smoothness
    try:
        res = minimize(objective, pts0.flatten(), method='Powell', 
                       options={'maxiter': 1000, 'maxfev': 5000})
        if time.time() < deadline:
            if res.fun < -best_score: # Wait, objective is -score. So lower is better.
                # res.fun is the minimum of -score, i.e., -max_score
                # So if res.fun is lower (more negative), the score is higher.
                # best_score was calculated as -objective.
                # Let's just compare -res.fun with best_score.
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_pts = res.x.reshape(N, 2)
    except Exception as e:
        pass
        
    # Optional: Try a second restart with small noise if time permits
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.05, (N, 2))
        pts1 = pts0 + noise
        try:
            res2 = minimize(objective, pts1.flatten(), method='Powell', 
                       options={'maxiter': 500, 'maxfev': 2000})
            current_score = -res2.fun
            if current_score > best_score:
                best_score = current_score
                best_pts = res2.x.reshape(N, 2)
        except Exception:
            pass
            
    return best_pts
