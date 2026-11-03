import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 50

def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _get_delaunay_edges(centers):
    """Get unique edges from Delaunay triangulation."""
    try:
        tri = Delaunay(centers)
        simplices = tri.simplices
        edges = set()
        for simplex in simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails
        return [(i, j) for i in range(N) for j in range(i+1, N)]

def _soft_min(values, k):
    """Compute a differentiable approximation of the minimum."""
    m = np.min(values)
    shifted = values - m
    # Clip to avoid overflow/underflow issues in exp
    shifted = np.clip(shifted, 0, 100.0/k) 
    exp_terms = np.exp(-k * shifted)
    log_sum = np.log(np.sum(exp_terms))
    return m - (1.0 / k) * log_sum

def _optimize_from_start(start_centers, bounds, max_iter=2000, k_start=100.0, k_end=1000.0):
    """
    Optimize centers using Delaunay-aware sparse objective with increasing sharpness.
    """
    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    def objective(params, k):
        centers = params.reshape(N, 2)
        centers = np.clip(centers, -0.5, 0.5)
        
        edges = _get_delaunay_edges(centers)
        
        active_dists = []
        
        for i, j in edges:
            d = np.linalg.norm(centers[i] - centers[j])
            active_dists.append(d / 2.0)
            
        wall_dists = 0.5 - np.abs(centers)
        active_dists.extend(wall_dists.flatten())
        
        active_dists = np.array(active_dists)
        
        if len(active_dists) == 0:
            return 0.0
            
        return -_soft_min(active_dists, k)

    # Multi-stage optimization
    stages = [
        (100.0, 500),
        (500.0, 500),
        (2000.0, 1000)
    ]
    
    current_centers = start_centers.copy()
    
    for k, iters in stages:
        try:
            result = minimize(
                objective, 
                current_centers.reshape(-1), 
                args=(k,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            centers = result.x.reshape(N, 2)
            centers = np.clip(centers, -0.5, 0.5)
            true_r = _calculate_true_radius(centers)
            
            if true_r > best_score:
                best_score = true_r
                best_centers = centers.copy()
                current_centers = centers.copy()
            else:
                # If it didn't improve, keep the previous best as the next start
                current_centers = best_centers.copy()
        except Exception:
            pass
            
    return best_centers, best_score

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid (best initial guess for circle packing)
    # Approximate spacing for r=0.07 is 0.14
    # We want to fit 50 points. 7x8 grid is 56, 7x7 is 49.
    # Let's try a slightly stretched hex grid.
    s = 0.145
    points = []
    # Generate a larger hex grid and crop
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    
    if len(points) >= N:
        # Sort by distance from center to pick the most central N points
        dists_to_center = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        inits.append(points[indices])
        
    # 2. Square Grid 7x7 + 1 (Center)
    # 7x7 grid has 49 points. Add one in the center of a cell?
    # Or just use 7x7 and let the optimizer move them.
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        # Take 49 from grid, add 1 slightly offset
        # Actually, just take 50 from a slightly larger grid
        pass # Handled by hex grid mostly, but let's add a specific square init
        
    # 3. Random Uniform (seeded)
    np.random.seed(42)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Jittered Hex
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.01, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
        
    # 5. 8x7 Grid (56 points) -> Pick 50
    grid_x8 = np.linspace(-0.49, 0.49, 8)
    grid_y7 = np.linspace(-0.49, 0.49, 7)
    grid87 = np.array([[x, y] for x in grid_x8 for y in grid_y7])
    if len(grid87) >= N:
        # Pick the 50 closest to center
        dists_to_center = np.linalg.norm(grid87, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        inits.append(grid87[indices])
        
    # 6. Randomized Grid
    np.random.seed(555)
    grid_x_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_y_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_points_rand = np.array([[x, y] for x in grid_x_rand for y in grid_y_rand])
    if len(grid_points_rand) >= N:
        indices = np.random.choice(len(grid_points_rand), N, replace=False)
        inits.append(grid_points_rand[indices])

    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    time_limit = 105.0 # Leave 15s buffer
    
    for i, init in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        # Allocate time based on remaining inits
        remaining_inits = len(inits) - i
        alloc_time = (time_limit - (time.time() - start_time)) / remaining_inits
        # Cap per-init time to prevent one bad init from eating all time
        alloc_time = min(alloc_time, 15.0)
        
        if alloc_time < 1.0:
            continue

        # Run optimization
        centers, score = _optimize_from_start(init, bounds)
        
        if score > best_score:
            best_score = score
            best_centers = centers
            
    # Final polish on the best result found
    if best_centers is not None and time.time() - start_time < time_limit - 5.0:
        # Run a high-precision optimization on the best result
        c_final, s_final = _optimize_from_start(best_centers, bounds, k_start=500.0, k_end=5000.0)
        if s_final > best_score:
            best_score = s_final
            best_centers = c_final

    if best_centers is None:
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
        
    # Ensure centers are strictly inside to avoid boundary issues in scoring if floating point errors occur
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
