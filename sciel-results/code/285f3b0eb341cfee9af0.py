import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def get_score(points):
        """Calculate the exact minimum radius for a given set of points."""
        if points is None:
            return 0.0
        points = np.asarray(points).reshape(-1, 3)
        if points.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        # Check bounds
        if np.any(np.abs(points) > 0.5 + 1e-9):
            return 0.0
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances
        wall_dists = 0.5 - np.abs(points)
        min_wall = np.min(np.min(wall_dists, axis=1))
        
        return min(min_pair, min_wall)

    def make_grid(nx, ny, nz):
        """Create a grid of points centered in the cube."""
        x = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        y = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        z = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        # Generate all points
        grid = np.array([(x, y, z) for z in z for y in y for x in x])
        # Shuffle to avoid bias in which points are kept if we truncate, 
        # but for 100 points in 5x5x4=100, we take all.
        # If we generate more, we should take the best subset? 
        # For now, just return the first N.
        return grid[:N]

    # Initial configurations
    inits = []
    
    # 5x5x4 Grid (100 points exactly)
    # x: 5, y: 5, z: 4
    # Spacing 0.2. Centered.
    # x: -0.4, -0.2, 0, 0.2, 0.4
    # y: -0.4, -0.2, 0, 0.2, 0.4
    # z: -0.3, -0.1, 0.1, 0.3
    x5 = np.linspace(-0.4, 0.4, 5)
    y5 = np.linspace(-0.4, 0.4, 5)
    z4 = np.linspace(-0.3, 0.3, 4)
    grid1 = np.array([(x, y, z) for z in z4 for y in y5 for x in x5])
    inits.append(grid1)
    
    # 4x5x5 Grid
    x4 = np.linspace(-0.3, 0.3, 4)
    y5b = np.linspace(-0.4, 0.4, 5)
    z5 = np.linspace(-0.4, 0.4, 5)
    grid2 = np.array([(x, y, z) for z in z5 for y in y5b for x in x4])
    inits.append(grid2)
    
    # 5x4x5 Grid
    x5c = np.linspace(-0.4, 0.4, 5)
    y4 = np.linspace(-0.3, 0.3, 4)
    z5b = np.linspace(-0.4, 0.4, 5)
    grid3 = np.array([(x, y, z) for z in z5b for y in y4 for x in x5c])
    inits.append(grid3)
    
    # Random jittered version of the best grid (grid1)
    # Add small random noise to escape local minima
    rng = np.random.default_rng(42)
    grid4 = grid1 + rng.normal(0, 0.02, grid1.shape)
    inits.append(grid4)
    
    # Additional perturbed grids
    grid5 = grid2 + rng.normal(0, 0.02, grid2.shape)
    inits.append(grid5)
    
    grid6 = grid3 + rng.normal(0, 0.02, grid3.shape)
    inits.append(grid6)

    best_score = 0.0
    best_points = inits[0] # Default to first grid

    # Update best score with initial grids
    for g in inits:
        s = get_score(g)
        if s > best_score:
            best_score = s
            best_points = g

    def objective(x_flat, beta=50.0):
        """
        Objective function to minimize.
        We want to maximize the minimum distance.
        Using soft-min approximation.
        """
        points = x_flat.reshape(-1, 3)
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e6)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(points)
        min_wall_per_point = np.min(wall_dists, axis=1)
        
        # Combine
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        all_dists = np.concatenate([pair_dists, min_wall_per_point])
        
        # Soft min
        # To prevent overflow/underflow, shift by max
        max_d = np.max(all_dists)
        
        shifted = all_dists - max_d
        # shifted <= 0
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return 0.0
        soft_min = max_d - (1.0 / beta) * np.log(sum_exp)
        
        return -soft_min

    bounds = [(-0.5, 0.5)] * (N * 3)

    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        
        try:
            # Stage 1: Coarse optimization with lower beta
            res1 = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 5000}
            )
            
            if time.time() > deadline:
                break
                
            points1 = res1.x.reshape(-1, 3)
            points1 = np.clip(points1, -0.5, 0.5)
            
            # Stage 2: Fine optimization with higher beta
            if time.time() < deadline - 10:
                res2 = minimize(
                    objective, 
                    points1.flatten(), 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 3000}
                )
                points = res2.x.reshape(-1, 3)
            else:
                points = points1
            
            points = np.clip(points, -0.5, 0.5)
            
            s = get_score(points)
            if s > best_score:
                best_score = s
                best_points = points
                
        except Exception:
            pass
    
    # Final polish pass on the best solution
    if time.time() < deadline - 5 and best_score > 0:
        try:
            res_polish = minimize(
                objective, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
            )
            points_polish = res_polish.x.reshape(-1, 3)
            points_polish = np.clip(points_polish, -0.5, 0.5)
            
            s_polish = get_score(points_polish)
            if s_polish > best_score:
                best_score = s_polish
                best_points = points_polish
        except Exception:
            pass

    return best_points
