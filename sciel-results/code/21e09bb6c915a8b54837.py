import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Strategy: Direct Search via Nelder-Mead with adaptive restarts.
        # This is a structurally different approach from CMA-ES (Evolution Strategy).
        # Nelder-Mead is a derivative-free simplex method that is very effective on smooth,
        # unimodal functions like Sphere, especially in low dimensions (dim=5).
        # It typically converges faster than CMA-ES in the early-to-mid stages of optimization
        # for such functions, which is crucial for AOCC scoring (early progress counts).
        
        # We use a multi-start strategy with Nelder-Mead.
        # Start points are chosen to cover the space, but we prioritize the center
        # since the optimum is guaranteed to be in [-4, 4]^5.
        
        # Generate deterministic start points
        starts = []
        # 1. Center
        starts.append(np.zeros(dim))
        
        # 2. Corners of the guaranteed region [-4, 4]^5 (sampled sparsely)
        # We can't sample all 32 corners, so we pick a few representative ones
        # and some random-ish points within the box.
        rng = np.random.default_rng(12345)
        
        # Add a few random points in [-4, 4]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, dim))
            
        # Add points on the axes
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 3.0
            starts.append(p)
            p[i] = -3.0
            starts.append(p)

        # Sort starts by distance to center? No, just iterate.
        # We will run Nelder-Mead from each start until budget is exhausted or time runs out.
        
        # Initial simplex size (step size)
        # For Nelder-Mead in scipy, we specify initial simplex via 'x0' and the method
        # usually constructs it with a default step size. We can control it by
        # manually constructing the simplex or using 'initial_simplex' if available?
        # scipy.optimize.minimize(method='Nelder-Mead') uses a default step size of 0.05 * ||x0|| or 0.00025.
        # This might be too small if x0 is 0. Let's use a wrapper or just trust it?
        # Actually, if x0 is 0, the step size is tiny. This is bad.
        # We should use a custom implementation or a method that allows step size control.
        # Or, we can shift the start points away from 0 slightly?
        # Better: Use COBYLA? Or just implement a simple Nelder-Mead?
        # Or use scipy with 'initial_simplex' option? No, that's not a standard option for Nelder-Mead in older scipy.
        # However, we can use `scipy.optimize.minimize` with `method='Nelder-Mead'` and it will work.
        # To handle the step size issue at x=0, we can start slightly off-center or use a larger step.
        # Let's use a custom Nelder-Mead implementation for full control and speed.
        
        def nelder_mead(func, x0, maxfev, bounds=(-5, 5)):
            """
            Simple Nelder-Mead implementation.
            """
            dim = len(x0)
            # Initial simplex
            simplex = []
            simplex.append(x0.copy())
            
            # Step size for initial simplex
            # If x0 is 0, use a default step like 0.5
            # If x0 is non-zero, use 0.05 * ||x0||
            norm_x0 = np.linalg.norm(x0)
            if norm_x0 < 1e-6:
                step = 0.5
            else:
                step = 0.05 * norm_x0
            
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                # Clip to bounds
                xi = np.clip(xi, bounds[0], bounds[1])
                simplex.append(xi)
            
            # Evaluate initial simplex
            fvals = []
            for x in simplex:
                f = func(x)
                fvals.append(f)
            
            # Sort simplex by function value
            def sort_simplex(simplex, fvals):
                idx = np.argsort(fvals)
                return [simplex[i] for i in idx], [fvals[i] for i in idx]
            
            simplex, fvals = sort_simplex(simplex, fvals)
            
            nfev = len(fvals)
            best_x = simplex[0].copy()
            best_f = fvals[0]
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            # Convergence criteria
            tol_f = 1e-12
            tol_x = 1e-12
            
            while nfev < maxfev:
                # Check convergence
                if abs(fvals[0] - fvals[-1]) < tol_f:
                    break
                if np.max(np.abs(np.array(simplex) - simplex[0])) < tol_x:
                    break
                
                # Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, bounds[0], bounds[1])
                f_r = func(x_r)
                nfev += 1
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, bounds[0], bounds[1])
                    f_e = func(x_e)
                    nfev += 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (simplex[-1] - centroid)
                    
                    x_c = np.clip(x_c, bounds[0], bounds[1])
                    f_c = func(x_c)
                    nfev += 1
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                            fvals[i] = func(simplex[i])
                            nfev += 1
                            if nfev >= maxfev:
                                break
                        
                        if nfev >= maxfev:
                            break
                
                # Re-sort
                simplex, fvals = sort_simplex(simplex, fvals)
                
                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                
                # Time check
                if time.time() > deadline - 5:
                    break
            
            return best_x, best_f, nfev

        # Run Nelder-Mead from each start
        # Allocate budget: give each start a portion of the budget.
        # Since Nelder-Mead is fast, we can try many starts.
        # Let's give each start 50 evaluations, up to a max of 10 starts.
        # Total budget 500.
        
        num_starts = min(len(starts), 10)
        evals_per_start = max(10, budget // num_starts)
        
        for i in range(num_starts):
            if time.time() > deadline - 5:
                break
            
            remaining_budget = budget - calls_made
            if remaining_budget <= 0:
                break
            
            # Use remaining budget for this start, but cap it to evals_per_start for diversity
            # Actually, if the first one converges fast, we should move to the next.
            # If it doesn't, we should give it more.
            # Let's just give it all remaining budget? No, that kills diversity.
            # Let's give it `evals_per_start` or remaining, whichever is smaller.
            # But if it's the last start, give it all.
            
            if i == num_starts - 1:
                maxfev = remaining_budget
            else:
                maxfev = min(evals_per_start, remaining_budget)
            
            try:
                x0 = starts[i]
                # Ensure x0 is within bounds
                x0 = np.clip(x0, -5, 5)
                
                x_best, f_best, n_evals = nelder_mead(func, x0, maxfev)
                
                calls_made += n_evals
                
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best
                
                # If we converged very well, we might stop early?
                # For Sphere, if f_best is very close to the minimum, we are done.
                # But we don't know f_opt.
                # If f_best is less than 1e-6, it's likely very close.
                # Let's check if we should stop.
                # Actually, let's just continue to use the budget for other starts
                # in case the first one got stuck in a local minimum (unlikely for Sphere)
                # or if the shift makes it look different.
                
            except Exception as e:
                # Fallback to random search if NM fails?
                # Or just skip.
                pass
        
        # If we have budget left, do a final local refinement from the best point
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            try:
                # Run NM again from the best point found so far with a smaller step?
                # Or just use the remaining budget for a more aggressive search?
                # Let's just run NM from the current best with the remaining budget.
                x_best, f_best, n_evals = nelder_mead(func, best_x, remaining)
                calls_made += n_evals
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best
            except Exception:
                pass

        return best_x, best_f
