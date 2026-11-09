import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_used = 1

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget or check_deadline():
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Coarse random sampling to find a good basin
        # Use about 10% of budget for initial exploration
        exploration_budget = max(10, int(budget * 0.1))
        
        # Start with origin
        # Already evaluated above
        
        for _ in range(exploration_budget - 1):
            if calls_used >= budget or check_deadline():
                break
            x_new = 5.0 * (rng.random(n) - 0.5)
            f_new = do_call(x_new)
            if f_new is None:
                break

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_used = 1

        # Nelder-Mead implementation
        def nelder_mead(x0, f0, max_iter=2000):
            nonlocal best_f, best_x, calls_used
            
            # Create initial simplex
            simplex = [x0.copy()]
            fvals = [f0]
            
            # Initial step size: small to start, but NM will expand if needed
            # For Rosenbrock, a moderate step is good. 0.5 is standard.
            initial_step = 0.5
            
            for i in range(n):
                x_new = x0.copy()
                x_new[i] += initial_step
                x_new = np.clip(x_new, -5, 5)
                if calls_used >= budget or check_deadline():
                    return
                f_new = do_call(x_new)
                if f_new is None:
                    return
                simplex.append(x_new)
                fvals.append(f_new)
            
            if len(simplex) < n + 1:
                return

            # NM Parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction (inside)
            sigma = 0.5   # Shrinkage
            
            ftol = 1e-12
            xtol = 1e-10
            
            for _ in range(max_iter):
                if calls_used >= budget or check_deadline():
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check convergence
                if fvals[-1] - fvals[0] < ftol:
                    break
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                
                f_r = do_call(x_r)
                if f_r is None:
                    break
                
                # Reflection is good
                if fvals[0] <= f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                # Expansion
                elif f_r < fvals[0]:
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    
                    f_e = do_call(x_e)
                    if f_e is None:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                # Contraction
                else:
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    
                    f_c = do_call(x_c)
                    if f_c is None:
                        break
                    
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    # Shrink
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            f_new = do_call(simplex[i])
                            if f_new is None:
                                return
                            fvals[i] = f_new
                        if check_deadline() or calls_used >= budget:
                            break

        # Phase 2: Polishing with Nelder-Mead
        # We will run NM on the best point found so far.
        # If it converges or stagnates, we might try another random point if budget allows.
        # However, NM is very efficient. Let's just run it on the current best.
        # To be safe against getting stuck in a bad local min, we could restart NM 
        # from a slightly perturbed best_x if it doesn't improve much.
        
        # Run NM from best_x
        nelder_mead(best_x.copy(), best_f)
        
        # If we still have budget, we can try a few more random restarts with NM
        # But NM is fast. Let's see how many calls we used.
        # If we used less than 50% of budget, we can afford more restarts.
        
        # A better strategy for BBOB: 
        # 1. Find a good point via random search.
        # 2. Polish it.
        # 3. If budget remains, try another random point, polish it.
        # 4. Keep the global best.
        
        # Let's implement a loop that alternates between random search and polishing
        # but prioritizes polishing if we are in a good region.
        
        # Actually, the previous logic did one NM run. 
        # Let's add a loop for multiple NM restarts if time/calls allow.
        
        while calls_used < budget and not check_deadline():
            # Pick a new start point
            # 50% chance to perturb current best, 50% random
            if rng.random() < 0.5 and calls_used > exploration_budget:
                x_start = best_x + 0.5 * rng.standard_normal(n)
            else:
                x_start = 5.0 * (rng.random(n) - 0.5)
            x_start = np.clip(x_start, -5, 5)
            
            f_start = do_call(x_start)
            if f_start is None:
                break
            
            # Run NM from this start
            # Limit calls per NM run to avoid wasting budget on a bad start
            # We can estimate remaining budget
            remaining = budget - calls_used
            if remaining < 2 * (n + 1):
                break
                
            # If the start is not significantly better than best, we might skip NM or limit it
            # But NM can still find a local min that is better.
            # Let's just run it.
            nelder_mead(x_start, f_start, max_iter=100)
            
            # If we are very close to the end, break
            if calls_used >= budget - 10:
                break

        return best_x, best_f
