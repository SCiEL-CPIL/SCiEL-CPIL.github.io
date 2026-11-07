"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Random Sampling + Nelder-Mead Simplex Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        # Track global best
        best_x = None
        best_f = float('inf')
        calls = 0

        # Helper to update best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Random Sampling to find promising basins
        # Use 20 initial samples to cover the space better for multimodal functions
        N_INIT = 20
        init_points = rng.uniform(-5, 5, (N_INIT, dim))
        
        # Evaluate all initial points
        init_fvals = []
        for i in range(N_INIT):
            x = init_points[i]
            f = func(x)
            calls += 1
            init_fvals.append(f)
            update_best(x, f)
            
            # Time check
            if time.time() > deadline - 10.0:
                break
        
        if best_x is None:
            # Fallback if something went wrong
            x = np.zeros(dim)
            f = func(x)
            calls += 1
            update_best(x, f)

        # Sort initial points by fitness
        sorted_indices = np.argsort(init_fvals)
        
        # Phase 2: Nelder-Mead Simplex Search
        # We will run NM on the best initial point.
        # If we find a better point during sampling (not really possible here as we sampled all),
        # or if we restart, we use the new best.
        
        # Define a simple Nelder-Mead implementation
        def nelder_mead(func, x0, f0, budget_limit, deadline, dim, rng, current_best_x, current_best_f):
            # Initialize simplex
            # Use a step size of 0.5 to explore quickly initially
            step = 0.5
            simplex = [x0.copy()]
            fvals = [f0]
            
            # Generate other vertices
            for i in range(dim):
                v = x0.copy()
                v[i] += step
                v = np.clip(v, -5, 5)
                fv = func(v)
                calls_local = 1
                simplex.append(v)
                fvals.append(fv)
                # Check if this vertex is better
                if fv < current_best_f:
                    current_best_f = fv
                    current_best_x = v.copy()
            
            # Main NM loop
            # We need to track calls carefully. 
            # The outer loop controls the budget.
            # This function will run until budget_limit or deadline.
            
            # To make it cleaner, we'll implement the logic inline or with a generator-like approach.
            # Given the constraints, let's implement a standard NM loop here.
            
            # We need to pass 'calls' and update it. Since we can't easily modify nonlocal 'calls' 
            # from within a nested function without 'nonlocal' (which we have), let's be careful.
            # Actually, 'calls' is in the enclosing scope. We can use 'nonlocal calls'.
            
            while calls < budget_limit:
                if time.time() > deadline - 1.0:
                    break
                
                # Sort simplex by fitness
                indices = np.argsort(fvals)
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]
                
                # Update global best
                if fvals[0] < current_best_f:
                    current_best_f = fvals[0]
                    current_best_x = simplex[0].copy()
                
                # Check for convergence
                # If the difference between best and worst is small, stop
                if abs(fvals[0] - fvals[-1]) < 1e-10:
                    break
                
                # Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls += 1
                
                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + 2 * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls += 1
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    # Outside contraction
                    x_con = centroid - 0.5 * (centroid - simplex[-1])
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls += 1
                    
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget_limit:
                                break
                        if calls >= budget_limit:
                            break
                        
                        # Update best after shrink
                        min_idx = np.argmin(fvals)
                        if fvals[min_idx] < current_best_f:
                            current_best_f = fvals[min_idx]
                            current_best_x = simplex[min_idx].copy()
                        
                        # If simplex is too small, restart or break
                        # Simple check: if max step is tiny
                        if np.max(np.abs(simplex[0] - simplex[1])) < 1e-8:
                            break

            return current_best_x, current_best_f

        # Run NM on the best initial point
        best_init_idx = sorted_indices[0]
        x_start = init_points[best_init_idx].copy()
        f_start = init_fvals[best_init_idx]
        
        # Run NM
        best_x, best_f = nelder_mead(func, x_start, f_start, budget, deadline, dim, rng, best_x, best_f)
        
        # If we still have budget, we could run NM on the 2nd best, etc.
        # But given the time constraints and that NM is effective, 
        # let's see if we have time.
        
        # If we have significant budget left, try the next best initial point
        if calls < budget - 10 and time.time() < deadline - 10:
            for idx in sorted_indices[1:]:
                if calls >= budget - 5:
                    break
                if time.time() > deadline - 5:
                    break
                
                x_try = init_points[idx].copy()
                f_try = init_fvals[idx]
                
                # Only run NM if this point is reasonably good
                # or if we have lots of budget left
                # For simplicity, just run it if we have budget.
                best_x, best_f = nelder_mead(func, x_try, f_try, budget, deadline, dim, rng, best_x, best_f)
                
                # Stop if we've used up budget
                if calls >= budget:
                    break

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            fx = func(x)
            calls += 1
            best_x = x
            best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
