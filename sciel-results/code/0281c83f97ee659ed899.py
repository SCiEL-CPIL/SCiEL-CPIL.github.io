import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 325.0  # Keep a 5s margin before the 330s kill

        rng = np.random.default_rng(42)
        
        # Phase 1: Initial Sampling
        # 12 random points to find a good basin
        n_init = 12
        X_init = rng.uniform(-5, 5, (n_init, dim))
        F_init = np.array([func(x) for x in X_init])
        
        best_idx = np.argmin(F_init)
        best_x = X_init[best_idx].copy()
        best_f = F_init[best_idx]
        calls = n_init
        
        # Track global best for AOCC (though we return best_x, best_f at the end, 
        # the evaluator records calls. We just need to minimize the function.)
        
        # We will use a multi-start Nelder-Mead approach.
        # Since we have 12 initial points, we can run NM from the top 2 best ones.
        # This balances exploration (multiple basins) and exploitation (local search).
        
        # Sort initial points by fitness
        sort_idx = np.argsort(F_init)
        top_k = 2
        starts = [X_init[sort_idx[i]] for i in range(top_k)]
        
        # Budget allocation:
        # We have budget - calls remaining.
        # We'll split the remaining budget roughly equally between the top_k starts,
        # but we can also do a dynamic allocation or just run them sequentially.
        # Given the time limit, we must be careful.
        
        # Nelder-Mead Implementation
        def nelder_mead(func, x0, budget_nm, dim, time_limit, start_time, rng):
            # Initialize simplex
            # Start with a small step size
            step = 0.3
            simplex = [x0.copy()]
            f0 = func(x0)
            calls = 1
            
            # Generate other vertices
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                simplex.append(xi)
            
            fvals = np.array([func(x) for x in simplex[1:]])
            calls += dim
            
            # Total simplex size is dim + 1
            # We maintain a list of (x, f) sorted by f
            
            # Helper to sort
            def sort_simplex(simplex, fvals):
                # Combine and sort
                combined = np.column_stack([np.array([fvals]), np.array(simplex)])
                # Actually, let's just use a list of tuples for clarity, 
                # or sort indices.
                # For efficiency, let's keep them in separate lists and sort indices
                pass
            
            # It's easier to maintain a list of points and sort it
            points = [(x, f) for x, f in zip(simplex, [f0] + list(fvals))]
            points.sort(key=lambda x: x[1])
            
            while calls < budget_nm:
                if time.time() - start_time > time_limit:
                    break
                
                # points[0] is best, points[-1] is worst
                x_worst = points[-1][0]
                f_worst = points[-1][1]
                x_best = points[0][0]
                f_best = points[0][1]
                
                # Centroid of all but worst
                centroid = np.mean([p[0] for p in points[:-1]], axis=0)
                
                # Reflection
                x_r = centroid + (centroid - x_worst)
                f_r = func(x_r)
                calls += 1
                
                if f_r < f_best:
                    # Expansion
                    x_e = centroid + 2 * (centroid - x_worst)
                    f_e = func(x_e)
                    calls += 1
                    
                    if f_e < f_r:
                        new_x, new_f = x_e, f_e
                    else:
                        new_x, new_f = x_r, f_r
                elif f_r < points[-2][1]:
                    # Accept reflection
                    new_x, new_f = x_r, f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                        f_c = func(x_c)
                        calls += 1
                        
                        if f_c <= f_r:
                            new_x, new_f = x_c, f_c
                        else:
                            # Shrink
                            new_points = []
                            for i, (x, f) in enumerate(points):
                                if i == 0:
                                    new_points.append((x, f))
                                else:
                                    x_s = x_best + 0.5 * (x - x_best)
                                    f_s = func(x_s)
                                    calls += 1
                                    new_points.append((x_s, f_s))
                            points = new_points
                            points.sort(key=lambda x: x[1])
                            continue
                    else:
                        # Inside contraction
                        x_c = centroid + 0.5 * (x_worst - centroid)
                        f_c = func(x_c)
                        calls += 1
                        
                        if f_c < f_worst:
                            new_x, new_f = x_c, f_c
                        else:
                            # Shrink
                            new_points = []
                            for i, (x, f) in enumerate(points):
                                if i == 0:
                                    new_points.append((x, f))
                                else:
                                    x_s = x_best + 0.5 * (x - x_best)
                                    f_s = func(x_s)
                                    calls += 1
                                    new_points.append((x_s, f_s))
                            points = new_points
                            points.sort(key=lambda x: x[1])
                            continue
                
                # Replace worst point
                points[-1] = (new_x, new_f)
                points.sort(key=lambda x: x[1])
                
                # Check convergence
                # If the best point hasn't improved significantly and the simplex is small, stop
                # Simple convergence check: spread of function values
                f_spread = points[-1][1] - points[0][1]
                x_spread = np.max(np.abs(np.array([p[0] for p in points]) - points[0][0]))
                
                if f_spread < 1e-10 and x_spread < 1e-8:
                    break
        
            return points[0][0], points[0][1], calls

        # Run NM from top starts
        remaining_budget = budget - calls
        budget_per_start = remaining_budget // top_k
        
        for i in range(top_k):
            if calls >= budget:
                break
            if time.time() - start_time > time_limit:
                break
            
            x_start = starts[i]
            # We already evaluated x_start in Phase 1, so we don't need to re-evaluate 
            # the initial point in NM, but my NM implementation does.
            # To save budget, we should ideally pass the initial f value.
            # However, for simplicity and since budget is 300, the overhead is small.
            # Let's just run it.
            
            x_opt, f_opt, calls_used = nelder_mead(func, x_start, budget_per_start, dim, time_limit, start_time, rng)
            calls += calls_used
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        
        return best_x, best_f
