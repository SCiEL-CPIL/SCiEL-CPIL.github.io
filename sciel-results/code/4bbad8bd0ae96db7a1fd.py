import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all combinations of 3 points
# C(14, 3) = 364
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guesses
    inits = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(x0)
    
    # 2. Grid 4x4 minus 2 corners
    # 16 points, remove (0,0) and (3,3)
    gx, gy = np.meshgrid(np.arange(4), np.arange(4))
    grid_pts = np.column_stack((gx.flatten(), gy.flatten()))
    # Remove first and last
    grid_pts = np.delete(grid_pts, [0, 15], axis=0)
    inits.append(grid_pts)
    
    # 3. Grid 3x5 minus 1 corner
    gx, gy = np.meshgrid(np.arange(3), np.arange(5))
    grid_pts = np.column_stack((gx.flatten(), gy.flatten()))
    grid_pts = np.delete(grid_pts, [0], axis=0) # 15 points
    inits.append(grid_pts)
    
    # 4. Random points in a circle
    rng = np.random.default_rng(42)
    r = rng.uniform(0, 1, size=(N, 1))
    theta = rng.uniform(0, 2*np.pi, size=(N, 1))
    x0 = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    inits.append(x0)
    
    best_points = inits[0]
    best_score = _score(best_points)
    
    # Objective function for minimization
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        return -_score(pts)
        
    # Run optimization
    # Powell is often better for non-smooth min functions
    # Nelder-Mead is also good. Let's try both or just Powell.
    # Powell can be slow but robust.
    
    for i, x_init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        # Try Powell
        try:
            res = minimize(objective, x_init.flatten(), method='Powell', 
                           options={'maxiter': 500, 'xtol': 1e-5, 'ftol': 1e-5})
            if res.success or res.fun < objective(x_init.flatten()):
                cand_pts = res.x.reshape(N, 2)
                cand_score = -res.fun
                if cand_score > best_score:
                    best_points = cand_pts
                    best_score = cand_score
        except Exception:
            pass
        
        # Try Nelder-Mead on the result of Powell if it improved
        if time.time() > deadline - 10.0:
            break
        try:
            res2 = minimize(objective, best_points.flatten(), method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res2.success or res2.fun < objective(best_points.flatten()):
                cand_pts = res2.x.reshape(N, 2)
                cand_score = -res2.fun
                if cand_score > best_score:
                    best_points = cand_pts
                    best_score = cand_score
        except Exception:
            pass

    return best_points
