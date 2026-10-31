import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2.0

# Precompute all triangle indices
tri_indices = list(combinations(range(N), 3))
NUM_TRI = len(tri_indices)  # 165

def triangle_areas_fast(points):
    """
    Calculate the areas of all triangles formed by the points using vectorized operations.
    """
    # points shape: (N, 2)
    # For each triangle (i, j, k), compute 0.5 * |cross(p_j - p_i, p_k - p_i)|
    i_idx = np.array([t[0] for t in tri_indices])
    j_idx = np.array([t[1] for t in tri_indices])
    k_idx = np.array([t[2] for t in tri_indices])
    
    p_i = points[i_idx]  # (NUM_TRI, 2)
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    # Cross product: (p_j - p_i) x (p_k - p_i)
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    return 0.5 * np.abs(cross)

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
    areas = triangle_areas_fast(points)
    min_area = np.min(areas)
    pen = penalty(points)
    return -min_area + 1000.0 * pen

def safe_objective(flat_params):
    points = flat_params.reshape(N, 2)
    if np.any(points[:, 0] < -1) or np.any(points[:, 0] > 2) or np.any(points[:, 1] < -1) or np.any(points[:, 1] > 2):
        return 1e6
    return objective(flat_params)

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Differential Evolution for global exploration
    # Use a smaller population and fewer iterations to fit in time
    bounds = [(0, 1), (0, H)] * N
    
    def de_objective(flat_params):
        points = flat_params.reshape(N, 2)
        areas = triangle_areas_fast(points)
        min_area = np.min(areas)
        pen = penalty(points)
        if pen > 1e-3:
            return -min_area + 1000.0 * pen
        return -min_area
    
    try:
        de_result = differential_evolution(
            de_objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=0.01,
            seed=42,
            polish=False,
            update_interval=1,
            init='sobol'
        )
        
        de_points = de_result.x.reshape(N, 2)
        de_min_area = np.min(triangle_areas_fast(de_points))
        de_pen = penalty(de_points)
        
        if de_pen < 1e-4:
            de_score = de_min_area
        else:
            de_score = de_min_area - 1000 * de_pen
            
        if de_score > best_score:
            best_score = de_score
            best_points = de_points.copy()
            
    except Exception:
        pass
    
    if time.time() - start_time > time_limit - 30:
        if best_points is not None:
            return best_points
        # Fallback to centroid
        return np.array([[0.5, H/3]] * N)
    
    # Strategy 2: Multiple Powell restarts from diverse initializations
    # Include the DE result, random points, and structured patterns
    
    initializations = []
    
    # 1. DE result
    if best_points is not None:
        initializations.append(best_points.copy())
    
    # 2. Random restarts
    rng = np.random.RandomState(0)
    for _ in range(10):
        init_pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            init_pts[i, 0] = v * 1.0 + w * 0.5
            init_pts[i, 1] = w * H
        initializations.append(init_pts)
    
    # 3. Structured: Grid-like points
    # Place points in a rough triangular grid
    grid_pts = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.25, H/2],
        [0.75, H/2],
        [0.5, H],
        [0.125, H/4],
        [0.375, H/4],
        [0.625, H/4],
        [0.875, H/4],
        [0.5, H/2]
    ])
    # Scale to fit inside unit triangle (already inside)
    initializations.append(grid_pts)
    
    # 4. Vertices + centroid + midpoints
    vert_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, H],
        [0.5, H/3],
        [0.25, H/6],
        [0.75, H/6],
        [0.5, H/2],
        [0.125, H/12],
        [0.375, H/12],
        [0.625, H/12],
        [0.875, H/12]
    ])
    initializations.append(vert_pts)
    
    for init_pts in initializations:
        if time.time() - start_time > time_limit - 10:
            break
        
        try:
            result = minimize(
                safe_objective,
                init_pts.flatten(),
                method='Powell',
                options={'maxiter': 200, 'maxfev': 20000, 'disp': False}
            )
            
            opt_points = result.x.reshape(N, 2)
            min_area = np.min(triangle_areas_fast(opt_points))
            pen = penalty(opt_points)
            
            if pen < 1e-4:
                score = min_area
            else:
                score = min_area - 1000 * pen
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        return np.array([[0.5, H/3]] * N)
        
    return best_points
