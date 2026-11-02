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
    deadline = start_time + 145.0 # Safety margin
    
    # Initial guess: Regular 14-gon
    # Centered at (0,0), radius 1
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = x0
    best_score = _score(best_points)
    
    # Objective function for minimization
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        return -_score(pts)
        
    # Run optimization
    # Nelder-Mead is good for this
    options = {
        'maxiter': 1000,
        'xatol': 1e-6,
        'fatol': 1e-6,
        'adaptive': True
    }
    
    try:
        res = minimize(objective, x0.flatten(), method='Nelder-Mead', options=options)
        if res.success or res.fun < objective(x0.flatten()):
            cand_pts = res.x.reshape(N, 2)
            cand_score = -res.fun
            if cand_score > best_score:
                best_points = cand_pts
                best_score = cand_score
    except Exception:
        pass
    
    # Check time
    if time.time() < deadline:
        # Try a few random restarts or perturbations
        # Perturb the regular polygon slightly
        rng = np.random.default_rng(42)
        for i in range(5):
            if time.time() > deadline - 5.0:
                break
                
            noise = rng.normal(0, 0.1, size=(N, 2))
            x_init = x0 + noise
            try:
                res = minimize(objective, x_init.flatten(), method='Nelder-Mead', options=options)
                if res.success or res.fun < objective(x_init.flatten()):
                    cand_pts = res.x.reshape(N, 2)
                    cand_score = -res.fun
                    if cand_score > best_score:
                        best_points = cand_pts
                        best_score = cand_score
            except Exception:
                pass

    return best_points
