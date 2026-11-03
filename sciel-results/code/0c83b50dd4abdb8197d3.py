"""Improved program using Delaunay-based sparse optimization."""
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
        # Fallback to all pairs if Delaunay fails (e.g., collinear points)
        return [(i, j) for i in range(N) for j in range(i+1, N)]

def _soft_min(dists, k):
    """Compute soft minimum of distances."""
    m = np.min(dists)
    shifted = dists - m
    exp_terms = np.exp(-k * shifted)
    log_sum = np.log(np.sum(exp_terms))
    return m - (1.0 / k) * log_sum

def _optimize_from_start(start_centers, bounds, max_time=20.0):
    """
    Optimization using Delaunay-aware sparse objective with multi-stage approach.
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
    
    # Stage 1: Coarse optimization with low k
    k_schedule = [100.0, 500.0, 2000.0, 10000.0]
    
    for k in k_schedule:
        if time.time() - max_time > 100.0:
            break
            
        try:
            result = minimize(
                objective, 
                best_centers.reshape(-1), 
                args=(k,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False}
            )
            centers = result.x.reshape(N, 2)
            centers = np.clip(centers, -0.5, 0.5)
            true_r = _calculate_true_radius(centers)
            
            if true_r > best_score:
                best_score = true_r
                best_centers = centers
        except Exception:
            pass
    
    # Local search: try small perturbations and re-optimize
    n_local_trials = 5
    for _ in range(n_local_trials):
        if time.time() - max_time > 100.0:
            break
        
        # Perturb the best solution
        noise_scale = 0.01
        perturbed = best_centers + np.random.normal(0, noise_scale, (N, 2))
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        try:
            result = minimize(
                objective, 
                perturbed.reshape(-1), 
                args=(2000.0,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False}
            )
            centers = result.x.reshape(N, 2)
            centers = np.clip(centers, -0.5, 0.5)
            true_r = _calculate_true_radius(centers)
            
            if true_r > best_score:
                best_score = true_r
                best_centers = centers
        except Exception:
            pass
    
    return best_centers, best_score

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid
    s = 0.14 
    points = []
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
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
        
    # 2. Square grid
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points)
        
    # 3. Random uniform
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Jittered hex
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.02, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
        
    # 5. Another random
    np.random.seed(1000)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 6. Randomized Grid
    np.random.seed(555)
    grid_x_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_y_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_points_rand = np.array([[x, y] for x in grid_x_rand for y in grid_y_rand])
    if len(grid_points_rand) >= N:
        indices = np.random.choice(len(grid_points_rand), N, replace=False)
        inits.append(grid_points_rand[indices])
    
    # 7. Concentric rings pattern
    ring_inits = []
    np.random.seed(777)
    center = np.array([0.0, 0.0])
    rings = [(0.0, 1), (0.15, 6), (0.30, 12), (0.42, 18), (0.49, 13)]
    for r, count in rings:
        if count == 1:
            ring_inits.append(center.copy())
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for angle in angles:
                ring_inits.append([r * np.cos(angle), r * np.sin(angle)])
    if len(ring_inits) >= N:
        inits.append(np.array(ring_inits[:N]))
    
    # 8. Spiral pattern
    spiral_inits = []
    np.random.seed(888)
    n_spiral = N
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n_spiral):
        r = 0.45 * np.sqrt(i / n_spiral)
        theta = i * golden_angle
        spiral_inits.append([r * np.cos(theta), r * np.sin(theta)])
    inits.append(np.array(spiral_inits))
    
    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    # Sort initializations by their initial radius (best first)
    init_scores = []
    for init in inits:
        init = np.clip(init, -0.49, 0.49)
        score = _calculate_true_radius(init)
        init_scores.append((score, init))
    init_scores.sort(key=lambda x: -x[0])
    
    for i, (init_score, init) in enumerate(init_scores):
        if time.time() - start_time > 105.0: 
            break
            
        remaining_time = 105.0 - (time.time() - start_time)
        max_time_for_init = min(remaining_time, 25.0)
        
        centers, score = _optimize_from_start(init, bounds, max_time=max_time_for_init)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    if best_centers is None:
        # Fallback
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
