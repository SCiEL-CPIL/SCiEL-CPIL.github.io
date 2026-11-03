import numpy as np
import time
import math
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

N = 300

def compute_score(points):
    """
    Compute the exact packing radius for a set of points.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    # Calculate pairwise distances
    # N=300, 300x300 matrix is fine
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair, min_bound)
    return score

def soft_min_value(points, k=100.0):
    """
    Compute the smooth minimum of pairwise distances and boundary distances.
    Returns a scalar value close to the min distance.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    # Extract upper triangle for pairs
    pair_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min: -1/k * log(sum(exp(-k * d)))
    # Shift by min to prevent underflow
    d_min = np.min(all_dists)
    # Clip d_min to avoid log(0) issues if d is exactly 0, though inf on diagonal handles pairs
    # bound_dists can be 0 if on boundary.
    exp_terms = np.exp(-k * (all_dists - d_min))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return d_min
    
    soft_min = d_min - (1.0 / k) * np.log(sum_exp)
    return soft_min

def build_neighbor_structure(points, k_neighbors=20):
    """
    Build a KD-tree and identify nearest neighbors for fast local evaluation.
    """
    tree = cKDTree(points)
    # Query for k_neighbors nearest neighbors for each point
    # dists[i, j] is distance from point i to its j-th neighbor
    # indices[i, j] is index of j-th neighbor
    dists, indices = tree.query(points, k=k_neighbors + 1)
    # Remove self (first column)
    neighbor_dists = dists[:, 1:]
    neighbor_indices = indices[:, 1:]
    return tree, neighbor_dists, neighbor_indices

def fast_soft_min(points, tree, neighbor_dists, neighbor_indices, k=100.0, 
                   global_min_cache=None):
    """
    Fast soft-min using neighbor tracking.
    Only considers distances to nearest neighbors + boundary.
    This is an approximation that's much faster than full O(N^2).
    """
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    # Use precomputed neighbor distances (these need updating when points move)
    # For now, recompute distances to neighbors
    n_neighbors = neighbor_indices.shape[1]
    pair_dists = np.empty((N, n_neighbors))
    for i in range(N):
        for j in range(n_neighbors):
            idx = neighbor_indices[i, j]
            pair_dists[i, j] = np.linalg.norm(points[i] - points[idx])
    
    # Flatten pair distances (avoid double counting by taking upper triangle conceptually)
    # But for soft-min, we can just use all neighbor distances
    all_dists = np.concatenate([pair_dists.flatten(), bound_dists])
    
    # Soft min
    d_min = np.min(all_dists)
    exp_terms = np.exp(-k * (all_dists - d_min))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return d_min
    
    soft_min = d_min - (1.0 / k) * np.log(sum_exp)
    return soft_min

def simulate_annealing_fast(points, time_budget):
    """
    Simulated Annealing with neighbor tracking for fast evaluation.
    """
    start_time = time.time()
    T = 0.015  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    
    current_points = points.copy()
    
    # Build initial neighbor structure
    tree, neighbor_dists, neighbor_indices = build_neighbor_structure(current_points, k_neighbors=15)
    
    # Initial score using fast method
    current_score = fast_soft_min(current_points, tree, neighbor_dists, neighbor_indices, k=80.0)
    
    best_points = current_points.copy()
    best_score = current_score
    
    step_size = 0.04
    rng = np.random.default_rng(42)
    
    # Adaptive step size based on temperature
    while time.time() - start_time < time_budget:
        if time.time() - start_time > time_budget:
            break
            
        # Pick a random point to move
        i = rng.integers(0, N)
        
        # Generate random perturbation
        delta = rng.normal(0, step_size * (0.5 + T / 0.015), size=2)
        
        old_pos = current_points[i].copy()
        new_points = current_points.copy()
        new_points[i] += delta
        
        # Project inside container
        rad = np.linalg.norm(new_points[i])
        if rad > 1.0:
            new_points[i] /= rad
            
        # Fast evaluation: only need to check distances to neighbors of i
        # and neighbors of i's neighbors
        # For simplicity, use the fast_soft_min which recomputes neighbor distances
        new_score = fast_soft_min(new_points, tree, neighbor_dists, neighbor_indices, k=80.0)
        
        delta_e = new_score - current_score
        
        # Acceptance criterion
        if delta_e > 0 or (T > 0 and rng.random() < np.exp(delta_e / T)):
            current_points = new_points
            current_score = new_score
            
            # Periodically rebuild neighbor structure (every ~1000 steps)
            # to keep it accurate
            if rng.integers(0, 100) == 0:
                tree, neighbor_dists, neighbor_indices = build_neighbor_structure(current_points, k_neighbors=15)
                
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            current_points[i] = old_pos  # Revert
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Reheat occasionally to escape local minima
            if rng.random() < 0.01:
                T = 0.005
                step_size = 0.03
                
    return best_points, best_score

def polish_least_squares(points, time_budget, k=200.0):
    """
    Polish using scipy.optimize.least_squares on the soft-min objective.
    """
    start_time = time.time()
    
    # R_target should be higher than any possible radius to ensure gradient points "up"
    # Max radius ~ 0.053. Use 0.1.
    R_target = 0.1 
    
    def residuals(x_flat):
        x = x_flat.reshape(N, 2)
        # Project inside container
        rad = np.linalg.norm(x, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            x = x.copy()
            x[mask] /= rad[mask, np.newaxis]
            
        s_val = soft_min_value(x, k)
        # Minimize (R_target - s_val)^2
        # This effectively maximizes s_val as long as s_val < R_target
        return np.array([R_target - s_val])

    x0 = points.flatten()
    
    try:
        result = least_squares(
            residuals, 
            x0, 
            method='trf', 
            max_nfev=5000, # Limit evals to control time
            xtol=1e-12, 
            ftol=1e-12
        )
        
        x_opt = result.x.reshape(N, 2)
        # Final check
        rad_opt = np.linalg.norm(x_opt, axis=1)
        if np.any(rad_opt > 1.0) or not np.all(np.isfinite(x_opt)):
             return points, compute_score(points)
             
        score_opt = compute_score(x_opt)
        
        if time.time() - start_time > time_budget:
             pass # Just log, we return best found
        
        return x_opt, score_opt
    except Exception as e:
        return points, compute_score(points)

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # s is the distance between centers
    # For hex packing, s = 2r
    # Approx r ~ 0.053, s ~ 0.106
    # Let's start with a slightly smaller s to ensure fit, then expand
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N)) * 0.95
    
    y_step = s_init * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_init) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_init / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_init + offset
            if np.hypot(x, y) < 0.99:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        # Keep the N closest to center? Or random?
        # Keeping closest to center is usually a good dense core
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        # Fill remaining with random points
        # This shouldn't happen with s_init chosen correctly, but just in case
        while len(points) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            points = np.vstack([points, p])
        points = points[:N]
        
    best_points = points.copy()
    best_score = compute_score(points)
    
    # 2. Global Optimization: Fast Simulated Annealing with neighbor tracking
    # Allocate 85% of time to SA
    sa_budget = (time_limit - (time.time() - start_time)) * 0.85
    sa_points, sa_score = simulate_annealing_fast(best_points, sa_budget)
    
    if sa_score > best_score:
        best_score = sa_score
        best_points = sa_points
        
    # 3. Local Optimization: Polish
    # Allocate remaining time to Polish
    polish_budget = time_limit - (time.time() - start_time) - 2.0
    if polish_budget > 1.0:
        polished_points, polished_score = polish_least_squares(best_points, polish_budget)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points
            
    # 4. Final Validation
    rad_final = np.linalg.norm(best_points, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_points)):
        return np.zeros((N, 2))
        
    return best_points
