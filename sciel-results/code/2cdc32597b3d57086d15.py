"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate min triangle area efficiently
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        diffs = points[i+1:] - pi
        m = len(diffs)
        if m < 2:
            continue
        
        # Compute cross products for all pairs in diffs
        # cross[a, b] = det(diffs[a], diffs[b])
        cross = diffs[:, 0][:, None] * diffs[:, 1][None, :] - diffs[:, 1][:, None] * diffs[:, 0][None, :]
        
        # We only care about a < b
        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        if not np.any(mask):
            continue
            
        areas = 0.5 * np.abs(cross[mask])
        current_min = np.min(areas)
        if current_min < min_area:
            min_area = current_min
            
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_score = _score(x0)
    best_points = x0.copy()
    
    # List of initial configurations to try
    configs = [
        x0,
        x0 + 0.1 * np.random.randn(N, 2),
        x0 + 0.05 * np.random.randn(N, 2),
        # Two rings: 7 outer, 7 inner
        np.vstack([
            np.column_stack([np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False)), 
                             np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False))]),
            np.column_stack([0.5*np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7), 
                             0.5*np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7)])
        ]),
        # Ellipse
        np.column_stack([1.2 * np.cos(angles), 0.8 * np.sin(angles)])
    ]
    
    for x_init in configs:
        if time.time() > deadline - 20.0:
            break
            
        try:
            # Use Nelder-Mead for robustness with non-smooth objective
            res = minimize(_neg_score, x_init.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p_opt = res.x.reshape(-1, 2)
            s_opt = _score(p_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_points = p_opt
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
