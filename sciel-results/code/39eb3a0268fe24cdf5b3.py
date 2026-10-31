import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2.0

# Pre-compute triangle indices
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_I = _triplets[:, 0]
_J = _triplets[:, 1]
_K = _triplets[:, 2]

def get_areas(points):
    """
    points: (N, 2)
    returns: (165,) array of areas
    """
    p1 = points[_I]
    p2 = points[_J]
    p3 = points[_K]
    
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    return 0.5 * np.abs(cross)

def boundary_penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    
    v1 = np.maximum(0, -d1)
    v2 = np.maximum(0, -d2)
    v3 = np.maximum(0, -d3)
    
    return np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2)

def objective(params, lambda_area, lambda_bound):
    alpha = params[-1]
    points = params[:22].reshape(N, 2)
    
    areas = get_areas(points)
    bp = boundary_penalty(points)
    
    violations = np.maximum(0, alpha - areas)
    ap = np.sum(violations**2)
    
    return -alpha + lambda_area * ap + lambda_bound * bp

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # We will try multiple seeds and a few lambda values
    # To improve convergence, we can also use a "warm start" from the best previous solution
    # but random restarts are safer for non-convex problems.
    
    for seed in range(30):
        if time.time() - start_time > time_limit:
            break
        
        rng = np.random.RandomState(seed)
        
        init_points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            init_points[i, 0] = v * 1.0 + w * 0.5
            init_points[i, 1] = w * H
        
        init_alpha = 0.01
        init_params = np.concatenate([init_points.flatten(), [init_alpha]])
        
        # Try different penalty strengths
        # Higher lambda_area makes the solution more feasible (alpha <= min_area)
        # but might make the optimization landscape steeper.
        for lam in [50.0, 100.0, 200.0]:
            if time.time() - start_time > time_limit:
                break
                
            try:
                res = minimize(
                    objective, 
                    init_params, 
                    method='L-BFGS-B', 
                    args=(lam, 1000.0),
                    options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                opt_params = res.x
                opt_points = opt_params[:22].reshape(N, 2)
                
                # Calculate true score
                areas = get_areas(opt_points)
                min_area = np.min(areas)
                bp = boundary_penalty(opt_points)
                
                if bp < 1e-8:
                    if min_area > best_score:
                        best_score = min_area
                        best_points = opt_points.copy()
                        
            except Exception:
                pass

    if best_points is None:
        return np.array([[0.5, H/3]] * N)
        
    return best_points
