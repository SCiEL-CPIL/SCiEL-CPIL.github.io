import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0 # Leave 5s margin for 150s limit

def compute_radius(centers):
    """Compute the exact radius for a given set of centers."""
    # Pairwise distances
    # Using broadcasting for efficiency
    # centers shape (N, 3)
    # diff shape (N, N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Cube is [-0.5, 0.5]
    # dist to wall for point c is min(0.5 - c_x, c_x + 0.5, ...)
    # Since c is in [-0.5, 0.5], 0.5 - c_x >= 0 and c_x + 0.5 >= 0
    # Actually, if c_x is in [-0.5, 0.5], the distance to the face x=0.5 is 0.5-c_x and to x=-0.5 is c_x+0.5.
    # The minimum of these is the distance to the nearest wall in that dimension.
    # Then take the min over all dimensions.
    
    # coords: (N, 3)
    # dist_to_pos_wall = 0.5 - coords
    # dist_to_neg_wall = coords + 0.5
    # min_dist_to_wall_per_dim = np.minimum(dist_to_pos_wall, dist_to_neg_wall) # (N, 3)
    # min_dist_to_wall = np.min(min_dist_to_wall_per_dim, axis=1) # (N,)
    
    # More efficient:
    # d_x = np.minimum(0.5 - centers[:,0], centers[:,0] + 0.5)
    # d_y = np.minimum(0.5 - centers[:,1], centers[:,1] + 0.5)
    # d_z = np.minimum(0.5 - centers[:,2], centers[:,2] + 0.5)
    # min_wall_dist = np.minimum(np.minimum(d_x, d_y), d_z)
    # min_wall = np.min(min_wall_dist)
    
    # Even simpler:
    # The distance from a point c to the boundary of [-0.5, 0.5]^3 is min(0.5 - max(c), min(c) + 0.5) ? No.
    # For each dimension k, the distance to the nearest boundary is min(0.5 - c_k, c_k + 0.5).
    # The distance to the boundary of the cube is the minimum of these distances over all 3 dimensions.
    
    d_x = np.minimum(0.5 - centers[:,0], centers[:,0] + 0.5)
    d_y = np.minimum(0.5 - centers[:,1], centers[:,1] + 0.5)
    d_z = np.minimum(0.5 - centers[:,2], centers[:,2] + 0.5)
    min_wall_dist = np.minimum(np.minimum(d_x, d_y), d_z)
    min_wall = np.min(min_wall_dist)
    
    return min(min_pair_dist / 2.0, min_wall)

def soft_min_radius(centers, alpha):
    """
    Compute a smooth surrogate for the negative radius.
    We want to minimize this.
    r = min(D) where D is the vector of all half-distances and wall distances.
    soft_min(D, alpha) = -1/alpha * log(sum(exp(-alpha * D)))
    We minimize -soft_min(D, alpha).
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    half_dists = dists / 2.0
    
    # Wall distances
    d_x = np.minimum(0.5 - centers[:,0], centers[:,0] + 0.5)
    d_y = np.minimum(0.5 - centers[:,1], centers[:,1] + 0.5)
    d_z = np.minimum(0.5 - centers[:,2], centers[:,2] + 0.5)
    wall_dists = np.minimum(np.minimum(d_x, d_y), d_z)
    
    # Combine all distances into one vector
    # Exclude inf values
    valid_half_dists = half_dists[half_dists < np.inf]
    all_dists = np.concatenate([valid_half_dists, wall_dists])
    
    if len(all_dists) == 0:
        return 0.0
        
    # Numerical stability: subtract max
    max_d = np.max(all_dists)
    # exp(-alpha * (d - max_d))
    # log(sum(exp(-alpha * d))) = log(sum(exp(-alpha * (d - max_d) * exp(alpha * max_d))))
    # = alpha * max_d + log(sum(exp(-alpha * (d - max_d))))
    # soft_min = -1/alpha * (alpha * max_d + log(sum(exp(-alpha * (d - max_d)))))
    # = -max_d - 1/alpha * log(sum(exp(-alpha * (d - max_d))))
    
    # We want to minimize -soft_min = max_d + 1/alpha * log(sum(exp(-alpha * (d - max_d))))
    
    shifted = -alpha * (all_dists - max_d)
    # Clip to prevent underflow/overflow issues in exp, though shifted <= 0
    # max shifted is 0. min shifted is -alpha * (max_d - min_d).
    # If alpha is large, this can be very negative, exp -> 0.
    
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    
    return max_d + log_sum_exp / alpha

def objective(params, alpha):
    """
    Objective function for scipy.optimize.
    params: flat array of length N*3
    """
    centers = params.reshape(-1, 3)
    # Add a small penalty if centers go out of bounds? 
    # The wall distance term in soft_min_radius handles this naturally if we allow 
    # centers to go outside, but the distance becomes negative?
    # d_x = min(0.5 - x, x + 0.5). If x > 0.5, 0.5-x is negative. 
    # So the "distance" becomes negative, which lowers the min, penalizing the solution.
    # This is good.
    return soft_min_radius(centers, alpha)

def construct():
    start_time = time.time()
    best_radius = 0.0
    best_centers = None
    
    # Initial guess: Random points in the cube
    # Or a grid? Random is better for escaping local minima if we do multiple restarts.
    # Let's try a few restarts with random initialization.
    
    rng = np.random.default_rng(42)
    
    num_restarts = 10 # We will adjust this based on time
    # We can also use a grid as one of the restarts
    
    for restart in range(num_restarts):
        if time.time() > DEADLINE - 10: # Stop if we have < 10s left
            break
            
        if restart == 0:
            # Start with a grid for a stable baseline
            m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            x0 = grid[:N].flatten()
        else:
            # Random initialization
            # Scale random points to be inside [-0.4, 0.4] to start safely inside
            x0 = rng.uniform(-0.4, 0.4, size=N * 3)
            
        # Optimization loop with increasing alpha
        # Start with low alpha for smoothness, increase for precision
        alphas = [5.0, 10.0, 20.0, 50.0, 100.0]
        
        current_params = x0.copy()
        
        for alpha in alphas:
            if time.time() > DEADLINE - 5:
                break
                
            # Run minimize
            # Use L-BFGS-B
            # maxiter should be reasonable to allow multiple restarts
            max_iter = 200 if alpha < 50 else 500
            
            try:
                res = minimize(
                    objective, 
                    current_params, 
                    args=(alpha,), 
                    method='L-BFGS-B', 
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_params = res.x
            except Exception as e:
                # If optimization fails, keep current params
                pass
                
        # Evaluate the final result
        centers = current_params.reshape(-1, 3)
        r = compute_radius(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
            # Optional: Polish step with very high alpha if time permits
            if time.time() < DEADLINE - 15 and r > 0.13:
                # Do a final polish with high alpha
                try:
                    res_polish = minimize(
                        objective, 
                        current_params, 
                        args=(500.0,), 
                        method='L-BFGS-B', 
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    centers_polished = res_polish.x.reshape(-1, 3)
                    r_polished = compute_radius(centers_polished)
                    if r_polished > best_radius:
                        best_radius = r_polished
                        best_centers = centers_polished.copy()
                except Exception:
                    pass

    if best_centers is None:
        # Fallback to grid if something went wrong
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
