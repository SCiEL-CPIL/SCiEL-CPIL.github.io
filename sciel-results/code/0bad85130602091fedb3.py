import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n = 8

    def compute_score(points):
        # points is shape (n, 2)
        # 1. Calculate pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-10)
        
        # 2. Find diameter
        mask = ~np.eye(n, dtype=bool)
        max_dist = np.max(dists[mask])
        
        if max_dist == 0:
            return -np.inf
            
        # 3. Compute log of product of ordered pair distances
        # Product is over i != j
        # log(product) = sum(log(dists[i, j])) for i != j
        log_prod_ordered = np.sum(np.log(dists[mask]))
        
        # 4. Objective to maximize:
        # K = log_prod_ordered - 56 * log(max_dist)
        # We minimize -K
        return log_prod_ordered - 56 * np.log(max_dist)

    def objective_full(x):
        pts = x.reshape(n, 2)
        s = compute_score(pts)
        if not np.isfinite(s):
            return 1e10
        return -s

    # Initial best: Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    p_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = compute_score(p_oct)
    best_points = p_oct.copy()
    
    # Check time
    if time.time() > deadline - 20:
        return [[float(x), float(y)] for x, y in best_points]

    # Strategy: Differential Evolution
    # DE is a population-based global optimizer that handles non-convexity well.
    # We define bounds for coordinates. 
    # Since the score is scale-invariant (normalized by diameter), 
    # the absolute scale doesn't matter, but the optimizer needs a bounded box.
    # A square of side 4 centered at origin is a reasonable search space.
    # If the optimal configuration is "flatter" or "taller", DE can find it within these bounds.
    # We can also try a larger box if needed, but 4x4 should cover the regular octagon 
    # (radius 1) with plenty of room for perturbations.
    
    bounds = [(-4.0, 4.0)] * (2 * n)
    
    # DE Parameters:
    # - popsize: 20 (total 20*16 = 320 evaluations per generation? No, popsize is multiplier of D)
    #   Actually, popsize=20 means 20*D individuals. D=16. So 320 individuals.
    #   That's too many for 120s? 
    #   Let's estimate speed. One compute_score is ~0.1ms? 
    #   320 evals * 0.1ms = 32ms per generation.
    #   maxiter=100 -> 3.2s. This is very fast.
    #   We can afford much larger popsize or more iterations.
    #   Let's try popsize=15, maxiter=200.
    
    # To improve reliability, we can run multiple DE instances with different seeds 
    # or use a "survival of the fittest" approach where we keep the best result.
    
    # We'll run one main DE instance first.
    try:
        # Use 'best1bin' strategy which is robust
        # tol and maxiter control convergence
        result = differential_evolution(
            objective_full, 
            bounds, 
            seed=42, 
            maxiter=300, 
            popsize=20, 
            tol=1e-8, 
            polish=True,  # Use L-BFGS-B on the best result to refine
            workers=1,
            init='sobol' # Sobol sequence for better coverage than random
        )
        
        if result.fun < -best_score:
            best_score = -result.fun
            best_points = result.x.reshape(n, 2)
            
    except Exception as e:
        pass

    # Check time
    if time.time() > deadline - 20:
        return [[float(x), float(y)] for x, y in best_points]

    # If DE didn't find anything significantly better, or if we have time,
    # try a second DE run with a different initialization strategy (random) 
    # to catch different basins of attraction.
    if time.time() < deadline - 30:
        try:
            result2 = differential_evolution(
                objective_full, 
                bounds, 
                seed=123, 
                maxiter=200, 
                popsize=15, 
                tol=1e-8, 
                polish=True,
                workers=1,
                init='latinhypercube'
            )
            
            if result2.fun < -best_score:
                best_score = -result2.fun
                best_points = result2.x.reshape(n, 2)
                
        except Exception as e:
            pass

    # Final local refinement from the best point found by DE
    if best_points is not None and time.time() < deadline - 5:
        try:
            # Use L-BFGS-B for high precision refinement
            res_refine = minimize(
                objective_full, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if res_refine.fun < -best_score:
                best_score = -res_refine.fun
                best_points = res_refine.x.reshape(n, 2)
                
        except Exception as e:
            pass

    # Ensure points are distinct (DE might collapse points if not careful, 
    # but log(0) penalty should prevent this. Still, add a safety check.)
    # If any points are too close, we might have an issue. 
    # However, the score function returns -inf for 0 distance, which DE treats as bad.
    
    return [[float(x), float(y)] for x, y in best_points]
