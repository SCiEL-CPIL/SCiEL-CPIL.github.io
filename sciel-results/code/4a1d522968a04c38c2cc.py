"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _soft_min(values, beta):
    """
    Compute the soft minimum of a set of values.
    values: array of shape (M,)
    beta: sharpness parameter
    """
    min_val = np.min(values)
    shifted = -beta * (values - min_val)
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return min_val
    return min_val - (1.0 / beta) * np.log(sum_exp)

def _get_constraints(centers):
    """
    centers: (N, 2)
    Returns:
      pair_dists_half: (N*(N-1)/2,)
      wall_dists: (N*2,) flattened
    """
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    pair_dists_half = dists[iu] / 2.0
    
    # Wall distances
    # dist to left: x - (-0.5) = x + 0.5
    # dist to right: 0.5 - x
    # dist to bottom: y + 0.5
    # dist to top: 0.5 - y
    # We want min of these 4 for each point, but the formula in task is:
    # min_i min_k (1/2 - |c_ik|)
    # This is equivalent to min(0.5 - |x|, 0.5 - |y|) for each point.
    # Let's compute 0.5 - |x| and 0.5 - |y| for all points.
    wall_dists = np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    return pair_dists_half, wall_dists

def _objective_flat(x, beta):
    """
    x: (2N,)
    Returns negative soft-min radius.
    """
    centers = x.reshape(N, 2)
    pair_dists_half, wall_dists = _get_constraints(centers)
    all_vals = np.concatenate([pair_dists_half, wall_dists.ravel()])
    sm = _soft_min(all_vals, beta)
    return -sm

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    pair_dists_half, wall_dists = _get_constraints(centers)
    min_pair = np.min(pair_dists_half)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _hard_min_polish(centers, max_iter=1000, step_size=1e-4):
    """
    Perform a simple subgradient ascent on the hard minimum radius.
    This helps correct any bias from the soft-min approximation.
    """
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    
    for _ in range(max_iter):
        # Compute pairwise distances and wall distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Find min pair distance
        min_pair_dist = np.min(dists)
        i_min, j_min = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Wall distances
        wall_dists = np.column_stack((
            0.5 - np.abs(centers[:, 0]),
            0.5 - np.abs(centers[:, 1])
        ))
        min_wall_dist = np.min(wall_dists)
        i_wall, j_wall = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        # Determine which constraint is active
        r_pair = min_pair_dist / 2.0
        r_wall = min_wall_dist
        
        if r_pair < r_wall:
            # Pair constraint active
            r = r_pair
            vec = diff[i_min, j_min]
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                continue
            grad = np.zeros_like(centers)
            grad[i_min] += vec / norm
            grad[j_min] -= vec / norm
        else:
            # Wall constraint active
            r = r_wall
            grad = np.zeros_like(centers)
            # wall_dists[i_wall, j_wall] is the min
            # If j_wall == 0, it's x-coord (0.5 - |x|). Grad wrt x is -sign(x)
            # If j_wall == 1, it's y-coord (0.5 - |y|). Grad wrt y is -sign(y)
            if j_wall == 0:
                grad[i_wall, 0] = -np.sign(centers[i_wall, 0])
                if centers[i_wall, 0] == 0: grad[i_wall, 0] = 1.0
            else:
                grad[i_wall, 1] = -np.sign(centers[i_wall, 1])
                if centers[i_wall, 1] == 0: grad[i_wall, 1] = 1.0
        
        # Normalize gradient
        norm_g = np.linalg.norm(grad)
        if norm_g < 1e-12:
            continue
        grad /= norm_g
        
        # Update
        new_centers = centers + step_size * grad
        # Clip to ensure validity (though gradient should push in)
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_r = _compute_radius(new_centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers.copy()
            centers = new_centers
            # Optionally decrease step size slightly
            step_size *= 0.99
        else:
            step_size *= 0.95
            if step_size < 1e-7:
                break
                
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    best_centers = None
    best_radius = -1.0
    
    # Initial guess: 7x7 grid (49 points) + 1 extra point
    # 7x7 grid spacing is 1/7. Radius ~ 0.0714.
    # 8x8 cropped grid spacing is 1/8. Radius ~ 0.0625.
    # 7x7 is much better start.
    
    def make_initial_grid_7x7_plus_1():
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid_49 = np.array([(x, y) for y in ticks for x in ticks])
        # Place 50th point. The center (0,0) is occupied.
        # Place it at the center of one of the cells, e.g., (0.5/7, 0)
        # Or just a random point near center.
        # Let's try placing it at (0.5/7, 0) which is midpoint between (0,0) and (1/7, 0)?
        # No, (0,0) and (1/7, 0) are adjacent. Midpoint is 1/14.
        # Distance to (0,0) is 1/14. Distance to (1/7, 0) is 1/14.
        # Radius would be limited by 1/14 / 2 = 1/28 ~ 0.035. Bad.
        # Better to place it in a hole. There are no holes in 7x7.
        # So we must distort the grid.
        # Let's just place it at (0, 0) + small random offset, and let optimizer move it.
        np.random.seed(42)
        extra = np.array([[0.0, 0.0]]) + np.random.uniform(-0.05, 0.05, size=(1, 2))
        centers = np.vstack([grid_49, extra])
        return centers

    initial_centers = make_initial_grid_7x7_plus_1()
    
    # Define bounds for optimizer
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Optimization settings
    # Start with low beta to explore, increase to sharpen
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0, 5000.0]
    
    restart_count = 0
    max_restarts = 5 # Limit restarts to ensure time for polish
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        
        # Generate initial configuration
        if restart_count == 1:
            x0 = initial_centers.flatten()
        else:
            # Jitter the grid
            jitter = 0.05 / (restart_count ** 0.5)
            x0 = initial_centers + np.random.normal(0, jitter, size=(N, 2))
            x0 = np.clip(x0, -0.49, 0.49)
            x0 = x0.flatten()
        
        current_x = x0
        current_radius = _compute_radius(current_x.reshape(N, 2))
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = current_x.reshape(N, 2).copy()
        
        # Run optimization stages
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
                
            try:
                res = opt.minimize(
                    _objective_flat, 
                    current_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_x = res.x
                current_radius = _compute_radius(current_x.reshape(N, 2))
                
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = current_x.reshape(N, 2).copy()
                    
            except Exception:
                break
        
        # If we found a very good radius, break early
        if best_radius > 0.07137:
             break

    # Final polish on the best result found using hard minimum subgradient
    if best_centers is not None and time.time() < deadline - 2.0:
        # Do a few rounds of hard min polish
        current_best = best_centers
        current_best_r = best_radius
        for _ in range(3):
            if time.time() > deadline - 1.0:
                break
            polished_centers, polished_r = _hard_min_polish(current_best, max_iter=2000, step_size=1e-4)
            if polished_r > current_best_r:
                current_best = polished_centers
                current_best_r = polished_r
                best_centers = current_best
                best_radius = current_best_r
            else:
                break

    if best_centers is None:
        return initial_centers
        
    # Ensure output is valid (finite, within bounds)
    # The problem states centers must be inside [-1/2, 1/2].
    # "A centre on or outside the square ... makes the candidate invalid."
    # So we must be strictly inside? Or on the boundary is invalid?
    # "inside the unit square" usually implies open set, but "on or outside" is explicitly invalid.
    # So we must clip to [-0.5 + eps, 0.5 - eps].
    eps = 1e-9
    best_centers = np.clip(best_centers, -0.5 + eps, 0.5 - eps)
    
    return best_centers
# EVOLVE-BLOCK-END
