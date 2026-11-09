import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We set a safety margin to return early.
    deadline = start_time + 140.0 

    def get_hard_score(points):
        """
        Calculate the exact minimum radius for a given set of points.
        Returns 0 if invalid.
        """
        if points is None:
            return 0.0
        points = np.asarray(points).reshape(-1, 3)
        if points.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        # Check bounds with a small tolerance to avoid floating point issues
        # The problem states centers must be inside [-0.5, 0.5]^3
        # Radius r must satisfy 0.5 - |c_ik| >= r
        if np.any(np.abs(points) > 0.5 + 1e-9):
            return 0.0
        
        # Pairwise distances
        # Use broadcasting for vectorization
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances
        wall_dists = 0.5 - np.abs(points)
        min_wall = np.min(np.min(wall_dists, axis=1))
        
        # The radius is the minimum of the two constraints
        return min(min_pair, min_wall)

    def make_jittered_grid(nx, ny, nz, noise_scale, rng):
        """
        Create a grid of points and add Gaussian noise.
        """
        # Ensure we get exactly N points by choosing grid dims that multiply to N
        # 5*5*4 = 100
        x_coords = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        y_coords = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        z_coords = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        grid = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        
        # Add noise
        noise = rng.normal(0, noise_scale, grid.shape)
        # Clip to stay within bounds initially to help optimizer
        grid_noisy = np.clip(grid + noise, -0.49, 0.49)
        return grid_noisy

    def objective_repulsive(x_flat, power=4.0, wall_weight=1.0):
        """
        Objective function to minimize.
        Minimizes sum of inverse powers of distances.
        This encourages points to spread out as much as possible.
        """
        points = x_flat.reshape(-1, 3)
        
        # Pairwise repulsion
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0) # Avoid division by zero, value doesn't matter as we ignore diagonal
        
        # We want to minimize 1/d^power. 
        # To avoid overflow for very small distances, we can clip distances from below.
        # However, if points coincide, the objective goes to infinity, which is good for minimization.
        # But numerical stability is a concern.
        dists_clipped = np.maximum(dists, 1e-6)
        pair_term = np.sum(1.0 / np.power(dists_clipped, power))
        
        # Wall repulsion
        # Distance to walls
        wall_dists = 0.5 - np.abs(points)
        # Clip to avoid division by zero if point is on wall
        wall_dists_clipped = np.maximum(wall_dists, 1e-6)
        wall_term = np.sum(1.0 / np.power(wall_dists_clipped, power)) * wall_weight
        
        return pair_term + wall_term

    def objective_soft_min(x_flat, beta=100.0):
        """
        Alternative objective: Soft min of distances.
        Maximize min distance -> Minimize -soft_min(distance)
        """
        points = x_flat.reshape(-1, 3)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e6)
        
        wall_dists = 0.5 - np.abs(points)
        min_wall_per_point = np.min(wall_dists, axis=1)
        
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        all_dists = np.concatenate([pair_dists, min_wall_per_point])
        
        # Soft min
        max_d = np.max(all_dists)
        shifted = all_dists - max_d
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return 0.0
        soft_min = max_d - (1.0 / beta) * np.log(sum_exp)
        
        return -soft_min

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate initial configurations
    inits = []
    
    # 1. Jittered 5x5x4 grids with different noise levels
    for noise in [0.01, 0.05, 0.1]:
        inits.append(make_jittered_grid(5, 5, 4, noise, rng))
        
    # 2. Random points
    for _ in range(2):
        pts = rng.uniform(-0.45, 0.45, (N, 3))
        inits.append(pts)

    # Pre-compute scores for inits to find a good starting point
    for init in inits:
        s = get_hard_score(init)
        if best_points is None or s > best_score:
            best_score = s
            best_points = init

    # Optimization Loop
    # We will try multiple optimizations from the best initial points and some random ones.
    # To save time, we limit the number of full optimizations.
    
    # Strategy: 
    # 1. Optimize the best initial point with Soft-Min (smooth enough with moderate beta)
    # 2. Refine with higher beta
    # 3. Repeat for a few other inits if time permits.

    candidates = []
    
    # Sort inits by their initial hard score (best first)
    scored_inits = [(get_hard_score(i), i) for i in inits]
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    max_optimizations = 5 # Reduced to ensure we finish in time
    
    for idx, (init_score, init_pts) in enumerate(scored_inits):
        if time.time() > deadline - 20: # Need 20s for final polish/return
            break
        if idx >= max_optimizations:
            break
            
        x0 = init_pts.flatten()
        
        try:
            # Stage 1: Soft Min Refinement with moderate beta
            # This directly targets the minimum distance constraint.
            res1 = minimize(
                objective_soft_min,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                args=(50.0,), # Moderate beta
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 2000}
            )
            
            if time.time() > deadline - 10:
                break
                
            points1 = res1.x.reshape(-1, 3)
            points1 = np.clip(points1, -0.5, 0.5)
            
            # Evaluate hard score after stage 1
            s1 = get_hard_score(points1)
            if s1 > best_score:
                best_score = s1
                best_points = points1
                
            # Stage 2: Soft Min Refinement with high beta
            # Sharper focus on the minimum distance.
            if time.time() < deadline - 10:
                res2 = minimize(
                    objective_soft_min,
                    points1.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    args=(200.0,), # Higher beta
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
                )
                
                if time.time() > deadline - 5:
                    break
                    
                points2 = res2.x.reshape(-1, 3)
                points2 = np.clip(points2, -0.5, 0.5)
                
                s2 = get_hard_score(points2)
                if s2 > best_score:
                    best_score = s2
                    best_points = points2
                    
        except Exception:
            pass

    # Final Polish
    if best_points is not None and time.time() < deadline - 2:
        try:
            # One last high-precision pass on the very best point found
            res_polish = minimize(
                objective_soft_min,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                args=(500.0,), # Very high beta for precision
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
            )
            
            points_polish = res_polish.x.reshape(-1, 3)
            points_polish = np.clip(points_polish, -0.5, 0.5)
            
            s_polish = get_hard_score(points_polish)
            if s_polish > best_score:
                best_score = s_polish
                best_points = points_polish
                
        except Exception:
            pass

    # If no valid points were found (should not happen), return a grid
    if best_points is None:
        best_points = make_jittered_grid(5, 5, 4, 0.01, rng)

    return best_points
