"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50
# EVOLVE-BLOCK-START

def _soft_objective(x):
    """
    Objective function to minimize.
    We want to maximize the 'soft minimum' of distances.
    Minus log-sum-exp of negative distances is a smooth approximation of min distance.
    Here we maximize sum of logs, so we minimize negative sum of logs.
    """
    # Reshape to (N, 3)
    pts = x.reshape(-1, 3)
    
    # Calculate pairwise distances
    # d_ij = ||p_i - p_j||
    # We need min_{i<j} d_ij and min_i min_k (0.5 - |p_ik|)
    
    # Pairwise distances
    # Using broadcasting for speed
    # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=-1)
    # This creates a N x N matrix.
    # We only need upper triangle i < j
    
    # Efficient way:
    # Compute all pairwise squared distances
    # (a-b)^2 = a^2 - 2ab + b^2
    sq_norms = np.sum(pts * pts, axis=1)
    sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * (pts @ pts.T)
    # Clip small negative values due to floating point errors
    sq_dists = np.maximum(sq_dists, 0)
    dists = np.sqrt(sq_dists)
    
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    # dist to wall = 0.5 - |coord|
    # min over 3 axes
    abs_pts = np.abs(pts)
    wall_dists = 0.5 - np.min(abs_pts, axis=1)
    
    # We want to maximize min(pair_dists) and min(wall_dists)
    # Approximate min by log-sum-exp:
    # min(x) approx -log(sum(exp(-beta * x))) / beta
    # For beta -> infinity, this approaches min(x).
    # We use a fixed beta or anneal it.
    # Let's use a moderate beta to keep it smooth but sensitive.
    beta = 10.0
    
    # Soft min of pair distances
    # sum(exp(-beta * d))
    exp_pair = np.exp(-beta * pair_dists)
    sum_exp_pair = np.sum(exp_pair)
    soft_min_pair = -np.log(sum_exp_pair) / beta
    
    # Soft min of wall distances
    exp_wall = np.exp(-beta * wall_dists)
    sum_exp_wall = np.sum(exp_wall)
    soft_min_wall = -np.log(sum_exp_wall) / beta
    
    # The effective radius is the min of these two
    # We can combine them: soft_min_total = min(soft_min_pair, soft_min_wall)
    # Or just sum of logs?
    # Sum of logs: sum(log(d_ij)) is not min.
    # The log-sum-exp trick above gives a smooth approximation of the MIN.
    # We want to maximize the minimum of the set of all constraints (pair and wall).
    # So we can take the min of the two soft-mins, or better, treat all distances (pairs and walls) as one set.
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    exp_all = np.exp(-beta * all_dists)
    sum_exp_all = np.sum(exp_all)
    soft_min_all = -np.log(sum_exp_all) / beta
    
    # We want to maximize soft_min_all, so minimize -soft_min_all
    return -soft_min_all

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -np.inf
    best_x = None
    
    # Number of restarts
    n_restarts = 20
    
    # Random seed for reproducibility in this specific run, but we want diversity
    # We'll just use np.random
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initialization: Random points inside [-0.4, 0.4] to start safely away from walls
        # Or use a jittered grid for better initial spread
        if i == 0:
            # First try: Jittered Grid
            # 3x3x3 grid is 27 points. We need 50.
            # Let's use a 4x4x4 grid (64 points) and randomly select 50?
            # Or just random. Random is often better for escaping bad symmetries if optimized well.
            # Let's try a mix.
            pts = np.random.uniform(-0.4, 0.4, (N, 3))
        else:
            # Subsequent tries: Random
            pts = np.random.uniform(-0.4, 0.4, (N, 3))
        
        x0 = pts.flatten()
        
        # Bounds: [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            # L-BFGS-B is good for this
            res = opt.minimize(
                _soft_objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            x_opt = res.x
            pts_opt = x_opt.reshape(N, 3)
            
            # Calculate actual score (raw min radius)
            # Pairwise distances
            sq_norms = np.sum(pts_opt * pts_opt, axis=1)
            sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * (pts_opt @ pts_opt.T)
            sq_dists = np.maximum(sq_dists, 0)
            dists = np.sqrt(sq_dists)
            iu = np.triu_indices(N, k=1)
            pair_dists = dists[iu]
            min_pair = np.min(pair_dists) / 2.0
            
            # Wall distances
            abs_pts = np.abs(pts_opt)
            wall_dists = 0.5 - np.min(abs_pts, axis=1)
            min_wall = np.min(wall_dists)
            
            radius = min(min_pair, min_wall)
            
            if radius > best_score:
                best_score = radius
                best_x = x_opt.copy()
                
        except Exception as e:
            pass

    # If we have time, do a local polish on the best solution
    if best_x is not None and time.time() < deadline - 5.0:
        # Use Nelder-Mead for final polish on the negative min-radius
        # Nelder-Mead doesn't support bounds well, so we must be careful
        # Or use L-BFGS-B again with a higher beta or different objective?
        # Let's try a simple gradient-free local search or just return.
        # The soft objective is usually very close to optimal for the min-radius.
        pass

    if best_x is None:
        # Fallback: Simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_x.reshape(N, 3)

# EVOLVE-BLOCK-END
