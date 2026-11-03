"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Place 50 equal non-overlapping circles in the unit square [-0.5, 0.5]^2
    to maximize the common radius.
    
    Uses Multi-Start L-BFGS-B optimization.
    """
    
    def objective(x):
        """
        Minimize negative maximum radius.
        x: shape (N*2,)
        """
        points = x.reshape(N, 2)
        
        # Pairwise distances
        # Use broadcasting for efficiency
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        # Exclude diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        # Distance to nearest wall
        dist_x = 0.5 - np.abs(points[:, 0])
        dist_y = 0.5 - np.abs(points[:, 1])
        min_boundary_dist = np.min(np.minimum(dist_x, dist_y))
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        # We want to maximize r, so minimize -r
        return -r

    def local_polish(points, time_budget, rng):
        """
        Simple hill climbing to polish the L-BFGS-B result.
        L-BFGS-B might stop when gradient is small but a small move could improve the min-distance.
        """
        current_score = -objective(points.ravel())
        best_points = points.copy()
        best_score = current_score
        step_size = 0.001
        start_time = time.time()
        
        while time.time() - start_time < time_budget:
            idx = rng.randint(N)
            # Try small random moves
            delta = rng.normal(0, step_size, size=2)
            new_points = points.copy()
            new_points[idx] += delta
            new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
            
            new_score = -objective(new_points.ravel())
            
            if new_score > current_score:
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                step_size = min(step_size * 1.05, 0.005)
            else:
                step_size *= 0.9
                if step_size < 1e-5:
                    step_size = 1e-5
                    
        return best_points, best_score

    start_time = time.time()
    time_limit = 115.0
    
    # Constraints for L-BFGS-B: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_points = None
    best_score = -np.inf
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Random uniform
    rng_init = np.random.RandomState(1)
    configs.append(rng_init.uniform(-0.45, 0.45, size=(N, 2)))
    
    # 2. Grid 10x5
    xs = np.linspace(-0.45, 0.45, 5)
    ys = np.linspace(-0.45, 0.45, 10)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = np.column_stack((X.ravel(), Y.ravel()))
    configs.append(grid_pts)
    
    # 3. Grid 5x10
    xs = np.linspace(-0.45, 0.45, 10)
    ys = np.linspace(-0.45, 0.45, 5)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = np.column_stack((X.ravel(), Y.ravel()))
    configs.append(grid_pts)
    
    # 4. Random with larger spread
    rng_init2 = np.random.RandomState(2)
    configs.append(rng_init2.uniform(-0.48, 0.48, size=(N, 2)))
    
    # 5. Clusters in corners to force spreading
    corners = np.array([[-0.4, -0.4], [0.4, -0.4], [-0.4, 0.4], [0.4, 0.4]])
    cluster_pts = []
    rng_init3 = np.random.RandomState(3)
    for i in range(N):
        c = corners[i % 4]
        offset = rng_init3.normal(0, 0.1, size=2)
        pt = c + offset
        pt = np.clip(pt, -0.48, 0.48)
        cluster_pts.append(pt)
    configs.append(np.array(cluster_pts))
    
    # Allocate time for L-BFGS-B runs
    # We have 5 configs. Let's try 5 seconds per config, then polish the best ones.
    
    num_configs = len(configs)
    
    for i, init_pts in enumerate(configs):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 5:
            break
            
        # Ensure init points are within bounds
        init_pts = np.clip(init_pts, -0.5, 0.5)
        
        # Run L-BFGS-B
        try:
            res = minimize(objective, init_pts.ravel(), method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 10000})
            
            if res.success or res.fun < best_score:
                cur_pts = res.x.reshape(N, 2)
                cur_score = -res.fun
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = cur_pts.copy()
        except Exception:
            continue
            
    # Polish the best solution found
    if best_points is not None:
        remaining = time_limit - 5 - (time.time() - start_time)
        if remaining > 0:
            rng_polish = np.random.RandomState(42)
            best_points, best_score = local_polish(best_points, remaining, rng_polish)
    else:
        # Fallback if all failed
        best_points = configs[0]
        best_score = -objective(best_points.ravel())
        
    return best_points

# EVOLVE-BLOCK-END
