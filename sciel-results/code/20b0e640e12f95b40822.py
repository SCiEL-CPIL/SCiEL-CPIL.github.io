import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        remaining_budget = self.budget
        
        # State variables
        best_x = None
        best_f = np.inf
        
        # Helper to check deadline
        def time_exceeded():
            return time.time() > deadline

        # Nelder-Mead implementation
        def nelder_mead(func, x0, budget, rng):
            # Initialize simplex
            # x0 is the best point. Create other vertices by perturbing x0.
            simplex = [x0.copy()]
            step_size = 0.5
            for i in range(dim):
                x_new = x0.copy()
                x_new[i] += step_size
                simplex.append(x_new)
            
            f_vals = [func(x) for x in simplex]
            calls_used = dim + 1
            
            # Track best
            best_idx = np.argmin(f_vals)
            best_x_nm = simplex[best_idx].copy()
            best_f_nm = f_vals[best_idx]
            
            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            
            while calls_used < budget and not time_exceeded():
                # Sort simplex by function value
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                # If the best point hasn't changed and we are stuck, break early?
                # We handle stagnation in the outer loop.
                
                x_worst = simplex[-1]
                x_second_worst = simplex[-2]
                x_best = simplex[0]
                
                # Compute centroid of all but the worst point
                centroid = np.sum(simplex[:-1], axis=0) / dim
                
                # 1. Reflection
                x_ref = centroid + alpha * (centroid - x_worst)
                f_ref = func(x_ref)
                calls_used += 1
                
                if f_ref < f_vals[0]:
                    # 2. Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = func(x_exp)
                    calls_used += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    # 3. Contraction
                    if f_ref < f_vals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    
                    f_con = func(x_con)
                    calls_used += 1
                    
                    if f_con < min(f_ref, f_vals[-1]):
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        # 4. Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            f_vals[i] = func(simplex[i])
                            calls_used += 1
                            if calls_used >= budget:
                                break
            
            # Find the best point in the final simplex
            final_order = np.argsort(f_vals)
            best_idx_final = final_order[0]
            return simplex[best_idx_final], f_vals[best_idx_final], calls_used

        # Phase 1: Multi-start to find a good basin
        # Use 5 starting points, each with 40 budget
        num_starts = 5
        budget_per_start = 40
        
        for i in range(num_starts):
            if time_exceeded():
                break
            if remaining_budget <= 0:
                break
            
            # Random start in [-4, 4] to avoid boundaries initially
            x0 = rng.uniform(-4, 4, dim)
            
            # Run Nelder-Mead
            x_res, f_res, calls_used = nelder_mead(func, x0, min(budget_per_start, remaining_budget), rng)
            remaining_budget -= calls_used
            
            if f_res < best_f:
                best_f = f_res
                best_x = x_res
        
        # Phase 2: Refine and Escape
        # Continue from the best point found.
        # Use adaptive restarts if stagnation occurs.
        
        if remaining_budget > 0 and best_x is not None:
            # Run a longer Nelder-Mead from the best point
            # We will monitor for stagnation.
            # Simple strategy: Run NM with remaining budget.
            # To handle multi-modality, we can periodically "kick" the best point.
            
            current_x = best_x.copy()
            current_f = best_f
            stagnation_count = 0
            kick_threshold = 15  # If no improvement for 15 calls, kick
            
            while remaining_budget > 0 and not time_exceeded():
                # Run NM for a chunk of budget, e.g., 20 calls
                chunk_budget = min(20, remaining_budget)
                
                x_res, f_res, calls_used = nelder_mead(func, current_x, chunk_budget, rng)
                remaining_budget -= calls_used
                
                if f_res < current_f:
                    current_x = x_res
                    current_f = f_res
                    best_f = current_f
                    best_x = current_x
                    stagnation_count = 0
                else:
                    # No improvement in this chunk
                    stagnation_count += chunk_budget
                    
                if stagnation_count >= kick_threshold:
                    # Apply a random kick to escape local minimum
                    # Perturb current_x by a random amount
                    kick_size = 1.0
                    kick = rng.uniform(-kick_size, kick_size, dim)
                    new_x = np.clip(current_x + kick, -5, 5)
                    
                    # Evaluate the kick
                    f_kick = func(new_x)
                    remaining_budget -= 1
                    
                    if f_kick < current_f:
                        current_x = new_x
                        current_f = f_kick
                        best_f = current_f
                        best_x = current_x
                        stagnation_count = 0
                    else:
                        # If kick didn't help, try a smaller kick or just reset stagnation count
                        # to allow more time in current basin?
                        # Or try a different direction.
                        # For simplicity, we just reset the counter to try again, 
                        # but this might loop. Let's try a smaller kick.
                        kick_size = 0.5
                        kick = rng.uniform(-kick_size, kick_size, dim)
                        new_x = np.clip(current_x + kick, -5, 5)
                        f_kick = func(new_x)
                        remaining_budget -= 1
                        if f_kick < current_f:
                            current_x = new_x
                            current_f = f_kick
                            best_f = current_f
                            best_x = current_x
                        stagnation_count = 0

        return best_x, best_f
