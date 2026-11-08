import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute indices for triangles
    i_idx, j_idx, k_idx = [], [], []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    def score(points):
        # points is shape (N, 2)
        # Calculate convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume # In 2D, volume is area
        except Exception:
            return -1.0 # Degenerate case
        
        if hull_area < 1e-10:
            return -1.0
            
        # Calculate triangle areas
        v1 = points[j_idx] - points[i_idx]
        v2 = points[k_idx] - points[i_idx]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        return min_area / hull_area

    def objective(points_flat):
        points = points_flat.reshape(N, 2)
        return -score(points) # Minimize negative score

    best_points = None
    best_score = -1.0
    
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Try a few restarts
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline - 20.0:
            break
            
        if r == 0:
            current_points = init_points.copy()
        else:
            # Add small noise
            current_points = init_points + 0.1 * np.random.randn(N, 2)
            # Re-center and scale to unit area? 
            # The score is invariant to translation and scaling?
            # Score = min_area / hull_area.
            # If we scale points by s, min_area scales by s^2, hull_area scales by s^2. Ratio is invariant.
            # If we translate, both are invariant.
            # So we don't strictly need to normalize, but it helps numerical stability.
            
        # Flatten for optimizer
        x0 = current_points.flatten()
        
        # Run optimization
        try:
            result = minimize(objective, x0, method='Powell', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            opt_points = result.x.reshape(N, 2)
            opt_score = -result.fun
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
                
        except Exception:
            pass
            
    if best_points is None:
        return init_points
        
    return best_points
