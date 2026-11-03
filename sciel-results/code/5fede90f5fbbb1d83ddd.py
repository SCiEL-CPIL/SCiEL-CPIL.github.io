"""
Sphere packing: 100 equal spheres in a unit cube [-0.5, 0.5]^3.
Combines FCC lattice initialization with Soft-Min L-BFGS-B optimization
and a critical-pair polishing phase to maximize the common radius.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
CUBE_LIMIT = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    if len(centers) < 2:
        return float(np.min(CUBE_LIMIT - np.abs(centers)))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    min_wall_dist = np.min(CUBE_LIMIT - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _softmin_objective_and_grad(x, beta):
    """
    Objective: minimize soft-min of distances (pairwise and wall).
    x: flattened centers (N*3,)
    beta: inverse temperature (larger beta = sharper min)
    Returns (-obj, -grad) for minimization.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances squared
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Wall distances
    wall_dists = CUBE_LIMIT - np.abs(centers)
    wall_sq_dists = wall_dists**2
    
    # Soft-min on squared pairwise distances
    finite_mask = np.isfinite(sq_dists)
    if np.any(finite_mask):
        max_sq = np.max(sq_dists[finite_mask])
        shifted = sq_dists - max_sq
        exp_terms = np.exp(-beta * shifted)
        exp_terms[~finite_mask] = 0.0
        sum_exp = np.sum(exp_terms)
        
        if sum_exp > 0:
            log_sum = np.log(sum_exp)
            soft_min_sq = max_sq - (1.0 / beta) * log_sum
        else:
            soft_min_sq = max_sq
    else:
        soft_min_sq = 0.0
        
    # Wall part
    max_wall_sq = np.max(wall_sq_dists)
    shifted_wall = wall_sq_dists - max_wall_sq
    exp_wall = np.exp(-beta * shifted_wall)
    sum_wall = np.sum(exp_wall)
    
    if sum_wall > 0:
        log_sum_wall = np.log(sum_wall)
        soft_min_wall_sq = max_wall_sq - (1.0 / beta) * log_sum_wall
    else:
        soft_min_wall_sq = max_wall_sq
        
    # Combine pair and wall soft-mins using soft-min
    vals = np.array([soft_min_sq, soft_min_wall_sq])
    max_val = np.max(vals)
    exp_vals = np.exp(-beta * (vals - max_val))
    sum_vals = np.sum(exp_vals)
    
    if sum_vals > 0:
        log_sum_vals = np.log(sum_vals)
        total_soft_min_sq = max_val - (1.0 / beta) * log_sum_vals
    else:
        total_soft_min_sq = max_val
        
    total_soft_min_sq = max(total_soft_min_sq, 0.0)
    obj_val = np.sqrt(total_soft_min_sq)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    if np.any(finite_mask) and sum_exp > 0:
        weights = exp_terms / sum_exp
        sum_w = np.sum(weights, axis=1)
        weighted_sum_x = weights @ centers
        
        # d(soft_min_sq_pair)/d(c_i) = sum_j weights[i,j] * (c_i - c_j)
        grad_pair_sq = sum_w[:, np.newaxis] * centers - weighted_sum_x
        
        # Wall gradient
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        d_wall_sq_dc = -2.0 * wall_dists * signs
        
        weights_wall = exp_wall / sum_wall
        grad_wall_sq = np.zeros_like(centers)
        for i in range(N):
            w = weights_wall[i]
            if w > 0:
                grad_wall_sq[i] += w * d_wall_sq_dc[i]
        grad_wall_sq *= -1.0 / beta
        
        # Combine gradients
        w_A = np.exp(-beta * (soft_min_sq - max_val)) / sum_vals
        w_B = np.exp(-beta * (soft_min_wall_sq - max_val)) / sum_vals
        
        combined_grad_sq = w_A * grad_pair_sq + w_B * grad_wall_sq
        
        if total_soft_min_sq > 1e-12:
            grad = combined_grad_sq / (2.0 * obj_val)
        else:
            grad = np.zeros_like(centers)
            
    return -obj_val, -grad

def _optimize_packing(init_centers, time_limit=10.0):
    """
    Optimize packing from initial centers using soft-min L-BFGS-B.
    """
    start_time = time.time()
    x0 = init_centers.flatten()
    
    # Beta schedule: start small (smooth), end large (sharp)
    betas = [10, 50, 100, 500, 1000, 2000, 5000]
    
    best_x = x0
    best_val = _compute_radius(init_centers)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        def obj_grad(x):
            if time.time() - start_time > time_limit:
                return 1e10, np.zeros_like(x)
            val, grad = _softmin_objective_and_grad(x, beta)
            if not np.all(np.isfinite(grad)):
                return 1e10, np.zeros_like(x)
            return val, grad
            
        try:
            result = minimize(
                obj_grad, x0,
                method='L-BFGS-B',
                bounds=[(-CUBE_LIMIT, CUBE_LIMIT)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            x0 = result.x
            r = _compute_radius(x0.reshape(N, 3))
            if r > best_val:
                best_val = r
                best_x = x0
        except Exception:
            pass
            
    return best_x.reshape(N, 3)

def _polish_critical_pairs(centers, time_limit=5.0):
    """
    Polish by focusing on the critical pairs (closest pairs and closest to walls).
    This acts as a local search to resolve tight constraints.
    """
    start_time = time.time()
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Find critical pairs
        tree = cKDTree(centers)
        # We look for pairs that are close to the current limiting distance
        # But simply taking the closest few is a good heuristic
        pairs = tree.query_pairs(0.25)  # Pairs within 0.25
        
        if len(pairs) == 0:
            break
            
        improved = False
        # Sort pairs by distance to prioritize the most critical ones
        pair_dists = [(i, j, np.linalg.norm(centers[i] - centers[j])) for i, j in pairs]
        pair_dists.sort(key=lambda x: x[2])
        
        for i, j, dist in pair_dists[:15]:  # Limit to 15 most critical pairs
            if time.time() - start_time > time_limit:
                break
                
            d = centers[i] - centers[j]
            if dist < 1e-10:
                continue
                
            direction = d / dist
            
            # Try moving i and j apart
            for step in [0.005, 0.002, 0.001, 0.0005]:
                new_centers = centers.copy()
                new_centers[i] += step * direction
                new_centers[j] -= step * direction
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                
                new_radius = _compute_radius(new_centers)
                if new_radius > best_radius:
                    centers = new_centers
                    best_radius = new_radius
                    improved = True
                    break
        
        if not improved:
            break
            
    return centers

def _fcc_init(n):
    """
    Generate an initial configuration based on a Face-Centered Cubic (FCC) lattice.
    FCC is the densest packing in infinite space. We generate a larger FCC grid
    and select the n points closest to the origin, then scale/shift to fit the cube.
    """
    # Generate FCC points
    # Basis vectors for FCC
    a = 1.0
    basis = [
        (0, 0, 0),
        (a/2, a/2, 0),
        (a/2, 0, a/2),
        (0, a/2, a/2)
    ]
    
    # Generate a grid of unit cells
    # We need enough points to select n=100 from the center
    # 5x5x5 unit cells * 4 points/cell = 500 points
    extent = 3 
    points = []
    for i in range(-extent, extent+1):
        for j in range(-extent, extent+1):
            for k in range(-extent, extent+1):
                for b in basis:
                    p = np.array([i*a + b[0], j*a + b[1], k*a + b[2]])
                    points.append(p)
                    
    points = np.array(points)
    
    # Calculate distance from origin
    dists = np.linalg.norm(points, axis=1)
    
    # Select n points closest to origin
    indices = np.argsort(dists)[:n]
    selected_points = points[indices]
    
    # Center the selected points
    center = np.mean(selected_points, axis=0)
    selected_points = selected_points - center
    
    # Scale to fit in the cube [-0.5, 0.5]
    # Find the max extent
    max_extent = np.max(np.abs(selected_points))
    if max_extent > 0:
        scale = 0.45 / max_extent  # Leave some margin for optimization
        selected_points = selected_points * scale
        
    return selected_points

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n, seed=None):
    """Generate random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # List of initializations
    inits = []
    
    # 1. FCC Lattice (Strongest candidate)
    inits.append(_fcc_init(N))
    
    # 2. Grid
    inits.append(_grid_init(N))
    
    # 3. Random inits
    for seed in range(5):
        inits.append(_random_init(N, seed=seed))
        
    # Optimize each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        try:
            time_remaining = deadline - time.time() - 5.0
            if time_remaining < 5.0:
                break
                
            # First optimize with gradient descent
            # Allocate more time to FCC and Grid
            opt_time = min(10.0, time_remaining * 0.6) if i < 2 else min(5.0, time_remaining * 0.4)
            optimized = _optimize_packing(init, time_limit=opt_time)
            r_opt = _compute_radius(optimized)
            
            # Final polish
            if time.time() < deadline - 5.0:
                polished = _polish_critical_pairs(optimized, time_limit=min(5.0, deadline - time.time() - 3.0))
                r_polished = _compute_radius(polished)
                if r_polished > r_opt:
                    r_opt = r_polished
                    optimized = polished
            
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
                
        except Exception:
            continue
            
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 3.0:
        try:
            polished = _polish_critical_pairs(best_centers, time_limit=min(5.0, deadline - time.time() - 2.0))
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()
        except Exception:
            pass
            
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
        
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
