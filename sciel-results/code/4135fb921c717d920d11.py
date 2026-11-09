"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Grid-Search + Nelder-Mead optimizer for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        total_budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_used = 0
        
        # Helper to update global best
        def update_best(x, f):
            nonlocal global_best_x, global_best_f, calls_used
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()
        
        # Phase 1: Coarse Global Search
        # For dim=5, a grid of 3^5 = 243 points is feasible within 500 budget.
        # This ensures we sample the entire [-5, 5]^5 space reasonably well.
        # We use 3 points per dimension: -5, 0, 5
        grid_points = np.array(np.meshgrid(*([-5.0, 0.0, 5.0] * dim))).T.reshape(-1, dim)
        
        # Shuffle grid points to avoid biasing towards corners if time runs out
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(grid_points))
        grid_points = grid_points[perm]
        
        # Evaluate grid points
        for x in grid_points:
            if calls_used >= total_budget:
                break
            if time.time() > deadline - 10.0:
                break
                
            fx = func(x)
            calls_used += 1
            update_best(x, fx)
            
            # If we have very good progress, we might stop early? 
            # No, let's use the rest for local refinement.
            
        # If we ran out of budget on grid, return best
        if calls_used >= total_budget or time.time() > deadline - 5.0:
            return global_best_x, global_best_f

        # Phase 2: Local Refinement with Nelder-Mead
        # Select the best point from the grid as the starting point for Nelder-Mead.
        # We can also try to run Nelder-Mead on the top few grid points if budget allows,
        # but usually the single best point is sufficient for a good local minimum.
        
        x0 = global_best_x.copy()
        
        # Nelder-Mead Parameters
        # Initial simplex
        simplex = [x0.copy()]
        # Add vertices offset from x0
        # Step size for initial simplex
        step = 0.5 
        for i in range(dim):
            v = x0.copy()
            # If x0 is near boundary, adjust step direction
            if v[i] > 4.0:
                v[i] -= step
            elif v[i] < -4.0:
                v[i] += step
            else:
                v[i] += step
            simplex.append(v)
        
        # Evaluate initial simplex
        simplex_f = []
        for i, v in enumerate(simplex):
            if calls_used >= total_budget:
                break
            if time.time() > deadline - 5.0:
                break
            fv = func(v)
            calls_used += 1
            update_best(v, fv)
            simplex_f.append(fv)
            
        if len(simplex_f) < dim + 1:
            return global_best_x, global_best_f

        # Nelder-Mead Loop
        # Standard parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        # Simplex is a list of points, simplex_f is list of function values
        # We maintain them sorted? Or just track indices?
        # For dim=5, simplex has 6 points.
        
        max_iterations = 200
        tol = 1e-8
        
        for _ in range(max_iterations):
            if calls_used >= total_budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Check convergence
            # If the spread of function values is small
            if simplex_f[-1] - simplex_f[0] < tol:
                break
            
            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            worst_point = simplex[-1]
            worst_f = simplex_f[-1]
            
            # 1. Reflection
            x_r = centroid + alpha * (centroid - worst_point)
            x_r = np.clip(x_r, -5, 5)
            
            if calls_used >= total_budget:
                break
            f_r = func(x_r)
            calls_used += 1
            update_best(x_r, f_r)
            
            if f_r < simplex_f[0]:
                # 2. Expansion
                if time.time() > deadline - 5.0:
                    break
                if calls_used >= total_budget:
                    break
                x_e = centroid + gamma * (centroid - worst_point)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_used += 1
                update_best(x_e, f_e)
                
                if f_e < f_r:
                    new_point = x_e
                    new_f = f_e
                else:
                    new_point = x_r
                    new_f = f_r
            elif f_r < simplex_f[-2]:
                # Reflection is good, but not better than best
                new_point = x_r
                new_f = f_r
            else:
                # 3. Contraction
                if time.time() > deadline - 5.0:
                    break
                if calls_used >= total_budget:
                    break
                # Outside contraction (if f_r is worse than worst, which is always true here by definition of worst)
                # Actually, standard NM: if f_r >= f_worst, do contraction.
                # Here f_r is already evaluated.
                # If f_r < f_worst but >= f_second_worst, we are in the "good" case above.
                # If f_r >= f_worst, we do contraction.
                # Wait, the condition `elif f_r < simplex_f[-2]` handles the case where it's better than second worst.
                # So we are in the case: f_r >= simplex_f[-2]. This includes f_r >= f_worst (impossible since worst is max) 
                # or f_second_worst <= f_r < f_worst.
                # In standard NM, if f_r >= f_worst, we contract towards centroid.
                # If f_r < f_worst but > f_second_worst, we also usually contract (inside contraction).
                
                # Inside contraction
                x_c = centroid + rho * (worst_point - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_used += 1
                update_best(x_c, f_c)
                
                if f_c < f_r:
                    new_point = x_c
                    new_f = f_c
                else:
                    # 4. Shrink
                    if time.time() > deadline - 5.0:
                        break
                    # Shrink all points except the best towards the best
                    best_point = simplex[0]
                    best_f = simplex_f[0]
                    
                    # We need to shrink the simplex. This requires evaluating dim points.
                    # If budget is tight, we might just break.
                    if calls_used + dim > total_budget:
                        break
                        
                    # Shrink
                    for i in range(1, len(simplex)):
                        if time.time() > deadline - 5.0:
                            break
                        if calls_used >= total_budget:
                            break
                        x_s = best_point + sigma * (simplex[i] - best_point)
                        x_s = np.clip(x_s, -5, 5)
                        f_s = func(x_s)
                        calls_used += 1
                        update_best(x_s, f_s)
                        simplex[i] = x_s
                        simplex_f[i] = f_s
                    # Replace worst with shrunk version? 
                    # Standard shrink replaces all non-best points.
                    # The loop above did that.
                    # We don't add a new point, we just updated the existing ones.
                    # So we continue to next iteration.
                    continue

            # Replace worst point with new point
            simplex[-1] = new_point
            simplex_f[-1] = new_f

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
