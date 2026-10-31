import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 11
SQRT3 = np.sqrt(3)

# Precompute triangle indices for all C(11,3) = 165 triplets
def _precompute_indices():
    indices = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                indices.append((i, j, k))
    idx = np.array(indices, dtype=np.int32)
    return idx[:, 0], idx[:, 1], idx[:, 2]

_I, _J, _K = _precompute_indices()

def triangle_areas_vectorized(points):
    """
    Calculate the areas of all 165 triangles formed by the 11 points.
    """
    p1 = points[_I]
    p2 = points[_J]
    p3 = points[_K]
    
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    pen = np.sum(np.maximum(0, -d1)**2) + np.sum(np.maximum(0, -d2)**2) + np.sum(np.maximum(0, -d3)**2)
    return pen

def objective(flat_params):
    points = flat_params.reshape(N, 2)
    min_area = np.min(triangle_areas_vectorized(points))
    pen = penalty(points)
    return -min_area + 1000.0 * pen

def safe_objective(flat_params):
    points = flat_params.reshape(N, 2)
    if np.any(points[:, 0] < -0.5) or np.any(points[:, 0] > 1.5) or np.any(points[:, 1] < -0.5) or np.any(points[:, 1] > 1.5):
        return 1e6
    return objective(flat_params)

def get_good_init(rng):
    """Generate a good initial guess using symmetric structure + perturbation."""
    h = SQRT3 / 2.0
    # Symmetric base configuration (11 points)
    # Center + rings with 3-fold symmetry
    pts = np.zeros((N, 2))
    
    # Option 1: Symmetric grid-like
    if rng.rand() < 0.3:
        # 1 center
        pts[0] = [0.5, h/3.0]
        # 3 at 1/3 height
        pts[1] = [0.25, h/6.0]
        pts[2] = [0.75, h/6.0]
        pts[3] = [0.5, h/2.0]
        # 3 at 2/3 height  
        pts[4] = [0.25, h/2.0]
        pts[5] = [0.75, h/2.0]
        pts[6] = [0.5, 2*h/3.0]
        # 3 near vertices
        pts[7] = [0.1, h/12.0]
        pts[8] = [0.9, h/12.0]
        pts[9] = [0.5, h/12.0]
        pts[10] = [0.5, 5*h/12.0]
    else:
        # Random but with some structure
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            pts[i, 0] = v * 1.0 + w * 0.5
            pts[i, 1] = w * h
    
    # Add small random perturbation
    pts += rng.randn(N, 2) * 0.02
    return pts

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    def evaluate(points):
        min_area = np.min(triangle_areas_vectorized(points))
        pen = penalty(points)
        if pen > 1e-3:
            return min_area - 1000 * pen
        return min_area
    
    # Stage 1: Multiple random restarts with Powell
    n_restarts = 80
    for seed in range(n_restarts):
        if time.time() - start_time > time_limit - 15:
            break
        
        rng = np.random.RandomState(seed)
        init_points = get_good_init(rng)
        flat_init = init_points.flatten()
        
        try:
            result = minimize(
                safe_objective, 
                flat_init, 
                method='Powell', 
                options={'maxiter': 150, 'maxfev': 15000, 'disp': False}
            )
            
            opt_points = result.x.reshape(N, 2)
            score = evaluate(opt_points)
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: symmetric configuration
        h = SQRT3 / 2.0
        best_points = np.array([
            [0.5, h/3.0],
            [0.25, h/6.0], [0.75, h/6.0], [0.5, h/2.0],
            [0.25, h/2.0], [0.75, h/2.0], [0.5, 2*h/3.0],
            [0.1, h/12.0], [0.9, h/12.0], [0.5, h/12.0],
            [0.5, 5*h/12.0]
        ])
        best_score = evaluate(best_points)
    
    # Stage 2: Basin hopping from best point
    if time.time() - start_time < time_limit - 15:
        try:
            bh_result = basinhopping(
                safe_objective,
                best_points.flatten(),
                minimizer_kwargs={'method': 'Powell', 'maxiter': 100, 'maxfev': 8000},
                niter=30,
                seed=42
            )
            bh_points = bh_result.x.reshape(N, 2)
            bh_score = evaluate(bh_points)
            
            if bh_score > best_score:
                best_score = bh_score
                best_points = bh_points.copy()
        except Exception:
            pass
    
    # Stage 3: Fine-tuning with multiple local optimizations from perturbed best
    if time.time() - start_time < time_limit - 5:
        for trial in range(5):
            if time.time() - start_time > time_limit - 3:
                break
            rng = np.random.RandomState(1000 + trial)
            perturbed = best_points + rng.randn(N, 2) * 0.01
            try:
                result = minimize(
                    safe_objective,
                    perturbed.flatten(),
                    method='Powell',
                    options={'maxiter': 100, 'maxfev': 10000, 'disp': False}
                )
                opt_points = result.x.reshape(N, 2)
                score = evaluate(opt_points)
                if score > best_score:
                    best_score = score
                    best_points = opt_points.copy()
            except Exception:
                pass
    
    return best_points
