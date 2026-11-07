import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = np.inf
        dim = self.dim

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                raise TimeoutError("Time limit exceeded")
            calls += 1
            if calls > self.budget:
                raise Exception("Budget exceeded")
            
            val = float(func(x))
            
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
                
            return val

        # Strategy: Multi-start Powell's Method
        # Powell's method is a direct search method that does not require gradients.
        # It is particularly effective for ill-conditioned problems like Schaffers F7
        # because it builds up a set of conjugate directions iteratively, which helps
        # navigate narrow valleys more effectively than simple coordinate descent or
        # fixed-direction methods like the current DFTR approach.
        
        # We will use a multi-start strategy:
        # 1. Generate a set of random starting points.
        # 2. For each starting point, run a local optimization using Powell's method.
        # 3. Keep the best result found so far.
        # 4. Stop when budget or time is exhausted.
        
        # Powell's Method Implementation (Simplified)
        # Standard Powell's method maintains a set of directions.
        # At each step, it minimizes along each direction.
        # If the total reduction is greater than twice the reduction along the largest single direction,
        # it replaces the worst direction with a new direction towards the point of maximum reduction.
        
        # To fit within the budget and time, we will implement a robust version
        # that reuses the best point from previous runs as a new starting point if it improves.
        
        # Initial random starts
        rng = np.random.default_rng(42)
        
        # Number of restarts. We'll let the budget decide.
        # Each Powell run consumes some evaluations.
        # Let's estimate: A Powell run in 5D might take ~50-100 evaluations to converge reasonably.
        # With 500 budget, we can afford ~5-10 restarts.
        
        # We'll structure the main loop as a series of Powell runs.
        # Each run starts from the current best point (or a random point initially).
        
        # To ensure exploration, the first few runs start from random points.
        # Subsequent runs start from the best point found so far, but with a small perturbation
        # to escape local minima if the previous run converged prematurely.
        
        # Powell's Method Core
        def powell_method(x0, f0, budget_left):
            # x0: starting point
            # f0: function value at x0
            # budget_left: max evaluations allowed for this run
            
            if budget_left < 10:
                return x0, f0, 0
            
            # Initialize directions with coordinate axes
            directions = [np.eye(dim)[i] for i in range(dim)]
            x = x0.copy()
            fx = f0
            evals_used = 0
            
            # Tolerance for stopping
            tol = 1e-10
            
            # Main loop
            while budget_left - evals_used > 2 * dim + 1 and time_ok():
                x_old = x.copy()
                fx_old = fx
                max_reduction = 0
                max_dir_idx = 0
                max_x = x.copy()
                
                # Minimize along each direction
                for i in range(dim):
                    d = directions[i]
                    # 1D minimization along x + t*d
                    # Use a simple golden section search or parabolic interpolation
                    # To save evaluations, we'll use a few steps of line search.
                    # We can estimate the step size based on previous steps or use a fixed initial step.
                    
                    # Initial step size
                    step = 0.1
                    # Try to find a better step size with a few evaluations
                    # We'll do a simple adaptive step:
                    # Evaluate at x + step*d, if better, increase step, else decrease.
                    # This is a bit heuristic. A more robust approach is to use scipy.optimize.minimize_scalar
                    # but we want to avoid heavy dependencies if possible, though scipy is available.
                    # Let's use scipy.optimize.minimize_scalar for robustness.
                    
                    from scipy.optimize import minimize_scalar
                    
                    def line_func(t):
                        nonlocal evals_used, budget_left
                        if evals_used >= budget_left:
                            return fx # Dummy value
                        x_new = x + t * d
                        x_new = np.clip(x_new, -5, 5)
                        val = wrapped_func(x_new)
                        evals_used += 1
                        return val
                    
                    # Bounds for t to stay within box roughly
                    # x_i + t*d_i in [-5, 5] => t in [-5-x_i, 5-x_i] / d_i
                    # Since d is a unit vector or coordinate axis, we can use a reasonable range.
                    # For coordinate axes, d_i is 1 or 0.
                    # Let's use a wider range to be safe, e.g., [-10, 10]
                    res = minimize_scalar(line_func, bounds=(-10, 10), method='bounded', options={'xatol': 1e-6})
                    
                    t_opt = res.x
                    x_new = x + t_opt * d
                    x_new = np.clip(x_new, -5, 5)
                    
                    # Ensure we have the function value. minimize_scalar returns fun value.
                    # But we need to be careful: if budget ran out, res.fun might be dummy.
                    # Let's re-evaluate if necessary? No, that wastes budget.
                    # If evals_used >= budget_left, we break.
                    if evals_used >= budget_left:
                        break
                    
                    fx_new = res.fun
                    reduction = fx - fx_new
                    
                    if reduction > max_reduction:
                        max_reduction = reduction
                        max_dir_idx = i
                        max_x = x_new.copy()
                    
                    x = x_new
                    fx = fx_new
                
                if evals_used >= budget_left:
                    break
                
                # Check for convergence
                if abs(fx_old - fx) < tol:
                    break
                
                # Update directions
                # If max_reduction is significant, replace the direction with the smallest reduction
                # with a new direction from x_old to x (which is x + max_x - x_old? No, x is the new point after all mins)
                # Actually, standard Powell replaces the direction corresponding to the smallest reduction
                # with the direction from the point before the cycle (x_old) to the point after the cycle (x).
                
                # Find direction with smallest reduction
                # We didn't store all reductions, just the max. We need to store them or re-evaluate?
                # Let's store reductions.
                # This requires modifying the loop above.
                
                # Given the complexity and budget constraints, let's simplify:
                # Just keep the coordinate directions? No, that's just coordinate descent.
                # Let's implement the direction update properly.
                
                # We'll re-do the loop storing reductions.
                # This is getting complex for a nested function.
                # Alternative: Use Nelder-Mead Simplex.
                # Nelder-Mead is easier to implement correctly and robustly for low dimensions.
                # It is a direct search method that maintains a simplex.
                # It works well for ill-conditioned problems if the initial simplex is sized correctly.
                
                # Let's switch strategy to Nelder-Mead for the local refinement.
                # It's a different design family (Simplex vs. Directional/Conjugate).
                
                return x, fx, evals_used

        # Revised Strategy: Multi-start Nelder-Mead
        # Nelder-Mead is a robust derivative-free method.
        # We will implement a robust Nelder-Mead with adaptive step size and restarts.
        
        def nelder_mead(x0, f0, budget_left):
            if budget_left < 2 * (dim + 1):
                return x0, f0, 0
            
            # Initialize simplex
            alpha = 1.0  # Step size for simplex expansion
            simplex = [x0.copy()]
            f_values = [f0]
            evals_used = 0
            
            # Add other vertices
            for i in range(dim):
                x_new = x0.copy()
                # Perturb along axis i
                # Determine step size based on box size
                # If x0[i] is close to boundary, step inwards
                if x0[i] > 0:
                    x_new[i] = x0[i] - alpha
                else:
                    x_new[i] = x0[i] + alpha
                x_new = np.clip(x_new, -5, 5)
                
                if evals_used >= budget_left:
                    break
                
                f_new = wrapped_func(x_new)
                evals_used += 1
                simplex.append(x_new)
                f_values.append(f_new)
            
            if evals_used >= budget_left:
                return np.array(simplex[np.argmin(f_values)]), min(f_values), evals_used
            
            # Main loop
            while budget_left - evals_used > 1 and time_ok():
                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                
                if evals_used >= budget_left:
                    break
                f_ref = wrapped_func(x_ref)
                evals_used += 1
                
                if f_ref < f_values[0]:
                    # Expansion
                    x_exp = centroid + 2 * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    
                    if evals_used >= budget_left:
                        break
                    f_exp = wrapped_func(x_exp)
                    evals_used += 1
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Reflection is good
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_values[-1]:
                        # Outside contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    
                    x_con = np.clip(x_con, -5, 5)
                    
                    if evals_used >= budget_left:
                        break
                    f_con = wrapped_func(x_con)
                    evals_used += 1
                    
                    if f_con < f_values[-1]:
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            if evals_used >= budget_left:
                                break
                            f_values[i] = wrapped_func(simplex[i])
                            evals_used += 1
                        
                        if evals_used >= budget_left:
                            break
                        
                        # Check for convergence (simplex size)
                        diff = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                        if np.linalg.norm(diff) < 1e-10:
                            break
            
            # Return best point
            best_idx = np.argmin(f_values)
            return simplex[best_idx], f_values[best_idx], evals_used

        # Multi-start loop
        # Start with a few random points, then refine the best one.
        
        # Phase 1: Random Sampling
        num_random = min(20, self.budget // 10)
        for i in range(num_random):
            if calls >= self.budget or not time_ok():
                break
            x = rng.uniform(-5, 5, size=dim)
            f = wrapped_func(x)
        
        # Phase 2: Local Refinement with Nelder-Mead
        # We will try to refine the best point found so far.
        # If it gets stuck, we might restart from a random point near it.
        
        # To be safe, we'll just run Nelder-Mead on the current best point.
        # If we have budget left, we could do more restarts, but Nelder-Mead can be slow.
        # Let's allocate the remaining budget to a single Nelder-Mead run on the best point.
        
        if calls < self.budget and time_ok():
            x_start = best_x.copy()
            f_start = best_f
            budget_left = self.budget - calls
            
            # Run Nelder-Mead
            x_opt, f_opt, evals_used = nelder_mead(x_start, f_start, budget_left)
            
            # Update best if improved
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt
        
        return best_x, best_f
