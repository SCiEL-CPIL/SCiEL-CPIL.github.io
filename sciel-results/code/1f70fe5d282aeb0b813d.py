"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area == 0:
            return 0.0
    except Exception:
        return 0.0
        
    # Compute minimum triangle area
    min_area = np.inf
    # Loop over all triples
    # N=13 -> 286 triples, very fast
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                # Cross product magnitude
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Define initial configurations
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_regular = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Regular 12-gon + Center
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    center = np.array([[0.0, 0.0]])
    init_center_ring = np.vstack([center, ring])
    
    # 3. Random initialization
    rng = np.random.default_rng(123)
    init_random = rng.random((N, 2))
    
    inits = [init_regular, init_center_ring, init_random]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Objective function to minimize (negative score)
        def objective(x):
            pts = x.reshape(N, 2)
            # Ensure points are not degenerate
            return -_score(pts)
            
        # Use Nelder-Mead for local optimization
        # It is robust for non-smooth functions
        try:
            # Limit iterations to prevent long runs on a single start
            # 1000 iterations of Nelder-Mead on 26D is fast
            res = minimize(objective, init.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(N, 2)
                
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline:
            break

    # If no valid point was found, fallback to regular 13-gon
    if best_points is None:
        best_points = init_regular
        
    return best_points
# EVOLVE-BLOCK-END
