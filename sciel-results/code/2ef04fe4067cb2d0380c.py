import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Global best tracking
        global_best_x = None
        global_best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, global_best_x, global_best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            # Clip to domain
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls += 1
            
            if f_val < global_best_f:
                global_best_f = f_val
                global_best_x = x_clipped.copy()
            
            return f_val

        rng = np.random.default_rng(42)

        # Strategy: Parallel Nelder-Mead Simplex Ensemble
        # 1. Global Exploration Phase to find diverse basins
        n_global = 30
        points = []
        points.append(np.zeros(dim))
        # Add corners of the search space for diversity
        for i in range(2**dim):
            corner = np.array([(-5.0 if (i >> j) & 1 else 5.0) for j in range(dim)])
            points.append(corner)
        # Add random points
        for _ in range(n_global - 1 - 2**dim):
            points.append(rng.uniform(-5, 5, dim))
            
        values = []
        for p in points:
            v = evaluate(p)
            values.append(v)
            
        if calls >= budget:
            return global_best_x, global_best_f

        # 2. Initialize Simplexes
        # We will run 5 independent Nelder-Mead simplexes.
        # We pick the 5 best points from the global exploration as starting points.
        # If we have fewer than 5 distinct good points, we use the best one and perturb it.
        
        # Sort points by value
        sorted_indices = np.argsort(values)
        
        n_simplexes = 5
        start_points = []
        for idx in sorted_indices[:n_simplexes]:
            start_points.append(points[idx].copy())
            
        # If we don't have enough unique starting points (though we should have 30+),
        # or if they are too close, we might need to adjust.
        # For now, assume the 5 best are distinct enough.
        
        # Initialize Simplexes
        # A simplex has dim+1 points.
        # We start with the best point and generate dim other points around it.
        simplexes = []
        for sp in start_points:
            simplex = [sp.copy()]
            # Generate initial simplex vertices
            # Use a small initial step size
            step_size = 0.5
            for j in range(dim):
                new_point = sp.copy()
                new_point[j] += step_size
                new_point = np.clip(new_point, -5.0, 5.0)
                simplex.append(new_point)
            
            # Evaluate the simplex points
            f_vals = []
            for p in simplex:
                f_vals.append(evaluate(p))
            
            simplexes.append({
                'points': simplex,
                'f_vals': f_vals,
                'calls_made': dim + 1
            })
            
        if calls >= budget:
            return global_best_x, global_best_f

        # 3. Nelder-Mead Loop
        # We interleave steps between simplexes to maintain balance.
        # Alpha, Gamma, Rho parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        eps = 1e-10
        
        # We will loop until budget is exhausted or all simplexes converge.
        active_simplexes = list(range(len(simplexes)))
        
        while calls < budget and len(active_simplexes) > 0:
            if time.time() > deadline:
                break
            
            # Process one step for each active simplex
            # To avoid bias, we can rotate which simplex goes first, or just iterate.
            # Since they are independent, iterating is fine.
            
            for idx in list(active_simplexes):
                if calls >= budget:
                    break
                
                simplex = simplexes[idx]
                points_list = simplex['points']
                f_vals = simplex['f_vals']
                
                # Sort points by fitness
                order = np.argsort(f_vals)
                points_list = [points_list[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # Check for convergence
                # If the spread is very small, consider it converged
                f_spread = f_vals[-1] - f_vals[0]
                if f_spread < 1e-10:
                    # Remove from active list
                    active_simplexes.remove(idx)
                    continue
                
                # Centroid of all points except the worst
                centroid = np.mean(points_list[:-1], axis=0)
                worst_point = points_list[-1]
                worst_f = f_vals[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst_point)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)
                
                if calls >= budget:
                    break
                
                # Check if reflection is better than worst
                if f_r < worst_f:
                    # Check if it's better than best (or second best)
                    if f_r < f_vals[0]:
                        # Expansion
                        x_e = centroid + gamma * (centroid - worst_point)
                        x_e = np.clip(x_e, -5.0, 5.0)
                        f_e = evaluate(x_e)
                        
                        if calls >= budget:
                            break
                        
                        if f_e < f_r:
                            # Use expansion
                            points_list[-1] = x_e
                            f_vals[-1] = f_e
                        else:
                            # Use reflection
                            points_list[-1] = x_r
                            f_vals[-1] = f_r
                    else:
                        # Use reflection
                        points_list[-1] = x_r
                        f_vals[-1] = f_r
                else:
                    # Reflection is worse, try contraction
                    # Inside contraction
                    x_c = centroid + rho * (worst_point - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = evaluate(x_c)
                    
                    if calls >= budget:
                        break
                    
                    if f_c < worst_f:
                        # Use contraction
                        points_list[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        # Move all points except best towards best
                        best_point = points_list[0]
                        for i in range(1, len(points_list)):
                            points_list[i] = best_point + 0.5 * (points_list[i] - best_point)
                            points_list[i] = np.clip(points_list[i], -5.0, 5.0)
                            f_vals[i] = evaluate(points_list[i])
                            
                            if calls >= budget:
                                break
                        
                        if calls >= budget:
                            break
                
                # Update simplex data
                simplex['points'] = points_list
                simplex['f_vals'] = f_vals
                simplex['calls_made'] += dim + 1 # Approximate, actual calls vary

        return global_best_x, global_best_f
