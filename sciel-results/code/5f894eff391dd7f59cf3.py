"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _min_triangle_area(pts):
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            v1 = pts[j] - pi
            for k in range(j + 1, n):
                v2 = pts[k] - pi
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = -np.inf
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Grid (4x4 minus 2)
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    inits.append(grid_pts[:14])
    
    # 3. Random (seeded)
    rng = np.random.RandomState(42)
    inits.append(rng.rand(N, 2) * 2 - 1)
    
    def objective(x):
        return -_score(x.reshape(-1, 2))
        
    for init_pts in inits:
        if time.time() > deadline:
            break
            
        current_score = _score(init_pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = init_pts.copy()
            
        # Optimize
        try:
            res = minimize(objective, init_pts.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if -res.fun > best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2)
        except Exception:
            pass
            
    if best_pts is None:
        return inits[0]
        
    return best_pts
# EVOLVE-BLOCK-END
