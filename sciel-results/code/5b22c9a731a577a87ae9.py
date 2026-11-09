import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 100

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. Safety margin of 10s.
    deadline = start_time + 140.0 

    def get_hard_score(points):
        """
        Calculate the exact minimum radius for a given set of points.
        """
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

    def objective_soft_min(x_flat, beta=100.0):
        """
        Objective function to minimize.
        Minimizes -softmin(distances) to maximize the minimum distance.
        """
        points = x_flat.reshape(-1, 3)
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e6)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(points)
        min_wall_per_point = np.min(wall_dists, axis=1)
        
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        all_dists = np.concatenate([pair_dists, min_wall_per_point])
        
        # Soft min calculation
        max_d = np.max(all_dists)
        shifted = all_dists - max_d
        # Clip to avoid overflow in exp if beta is very high and distances vary a lot
        # But shifted <= 0, so exp(-beta * shifted) = exp(positive). 
        # If beta is high, terms where d << max_d will be 0. Terms where d ~ max_d will be 1.
        # To be safe against overflow if beta*|shifted| is large:
        # Actually, shifted is negative. -beta * shifted is positive.
        # If max_d is 0.2 and min_d is 0.1, shifted is -0.1. -beta * -0.1 = 0.1 * beta.
        # If beta=100, exp(10) ~ 22000. Sum of 4950 terms could be large but manageable.
        # If beta=1000, exp(100) is huge. 
        # We should normalize or use a safer beta for high precision, or rely on the fact that 
        # only the minimum few matter.
        
        # A safer soft min:
        # logsumexp trick: max_d - 1/beta * log(sum(exp(-beta * (d - max_d))))
        # Since d <= max_d, (d - max_d) <= 0.
        # Let u = max_d - d >= 0.
        # sum(exp(-beta * (d - max_d))) = sum(exp(beta * u))
        # This can overflow.
        
        # Better: 
        # soft_min = max_d - (1/beta) * log( sum( exp( -beta * (d_i - max_d) ) ) )
        # Note that -beta * (d_i - max_d) = beta * (max_d - d_i).
        # Let m = min(all_dists).
        # The term for the minimum distance will have exponent 0 (if we shift by min).
        # Let's shift by min_d to avoid overflow.
        min_d = np.min(all_dists)
        shifted_from_min = all_dists - min_d # >= 0
        exp_terms = np.exp(-beta * shifted_from_min) # <= 1, no overflow
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return 0.0
        soft_min = min_d - (1.0 / beta) * np.log(sum_exp)
        
        return -soft_min

    def make_grid(nx, ny, nz):
        x = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        y = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        z = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        grid = np.array([(x, y, z) for z in z for y in y for x in x])
        return grid[:N]

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # 1. Generate Initial Candidates
    inits = []
    
    # Standard Grid 5x5x4
    grid_554 = make_grid(5, 5, 4)
    inits.append(grid_554)
    
    # Grid 4x5x5
    grid_455 = make_grid(4, 5, 5)
    inits.append(grid_455)
    
    # Grid 5x4x5
    grid_545 = make_grid(5, 4, 5)
    inits.append(grid_545)
    
    # Jittered Grid
    grid_jittered = grid_554 + rng.normal(0, 0.02, grid_554.shape)
    grid_jittered = np.clip(grid_jittered, -0.49, 0.49)
    inits.append(grid_jittered)
    
    # Random
    random_pts = rng.uniform(-0.45, 0.45, (N, 3))
    inits.append(random_pts)
    
    # Evaluate initial scores
    for init in inits:
        s = get_hard_score(init)
        if best_points is None or s > best_score:
            best_score = s
            best_points = init

    # 2. Global Optimization: Differential Evolution
    # We run DE on the best initial point's neighborhood or from scratch?
    # DE is stochastic. Let's run it once with a reasonable budget.
    # To save time, we can run DE with a smaller population and fewer iterations,
    # or run it on the flattened vector.
    
    if time.time() < deadline - 30: # Need 30s for local refinement
        try:
            # Use a smoother objective for DE to avoid getting stuck in noisy local minima
            # beta=50 is a good balance
            de_result = differential_evolution(
                objective_soft_min, 
                bounds, 
                args=(50.0,),
                maxiter=50, 
                popsize=20, 
                tol=1e-5,
                seed=42,
                init='latinhypercube',
                polish=False # We will do our own polishing
            )
            
            if de_result.success or de_result.fun < -best_score: # -fun is approx soft_min
                de_points = de_result.x.reshape(-1, 3)
                de_points = np.clip(de_points, -0.5, 0.5)
                s_de = get_hard_score(de_points)
                if s_de > best_score:
                    best_score = s_de
                    best_points = de_points
        except Exception:
            pass

    # 3. Local Refinement: L-BFGS-B
    # Refine the best point found so far (either from inits or DE)
    # We can also try refining the top 2-3 distinct candidates if time permits.
    
    candidates_to_refine = []
    if best_points is not None:
        candidates_to_refine.append(best_points)
    
    # Also add the best initial grid if it wasn't the best overall
    if grid_554 is not best_points and get_hard_score(grid_554) > 0.1:
        candidates_to_refine.append(grid_554)
        
    # If DE found a good point, add it (already in best_points, but if best_points was grid, DE might be 2nd best)
    # Let's just refine the global best and the best grid.
    
    for cand in candidates_to_refine:
        if time.time() > deadline - 10:
            break
            
        try:
            # Stage 1: Moderate beta for stability
            res1 = minimize(
                objective_soft_min,
                cand.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                args=(100.0,),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 5000}
            )
            
            if time.time() > deadline - 5:
                continue
                
            points1 = res1.x.reshape(-1, 3)
            points1 = np.clip(points1, -0.5, 0.5)
            
            s1 = get_hard_score(points1)
            if s1 > best_score:
                best_score = s1
                best_points = points1
                
            # Stage 2: High beta for precision
            if time.time() < deadline - 2:
                res2 = minimize(
                    objective_soft_min,
                    points1.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    args=(500.0,),
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
                )
                
                points2 = res2.x.reshape(-1, 3)
                points2 = np.clip(points2, -0.5, 0.5)
                
                s2 = get_hard_score(points2)
                if s2 > best_score:
                    best_score = s2
                    best_points = points2
                    
        except Exception:
            pass

    # Final safety check
    if best_points is None:
        best_points = grid_554

    return best_points
