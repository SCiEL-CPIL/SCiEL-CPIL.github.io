"""Model-and-optimize optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES optimizer with restarts for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_1plus_lambda_es(self, func, x0, sigma0, budget, rng, deadline):
        """
        Run (1+lambda)-ES from x0 with step size sigma0.
        Returns (best_x, best_f, evals_used).
        """
        dim = self.dim
        lam = max(4, 4 * dim)  # Population size (number of offspring)
        
        x_best = np.array(x0, dtype=float)
        f_best = func(x_best)
        evals = 1
        sigma = sigma0
        
        # 1/5 rule parameters
        success_threshold = 0.2
        sigma_min = 1e-8
        sigma_max = 10.0
        
        no_improve = 0
        stall_threshold = 20 # Generations without improvement
        
        while evals < budget and time.time() < deadline:
            # Generate offspring
            offspring = []
            for _ in range(lam):
                if evals >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x_best + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                offspring.append((y, fy))
            
            if not offspring:
                break
                
            # Find best offspring
            best_offspring = min(offspring, key=lambda t: t[1])
            
            # Update best
            if best_offspring[1] < f_best:
                x_best = best_offspring[0]
                f_best = best_offspring[1]
                no_improve = 0
                # Increase step size on success (1/5 rule)
                # Count successes
                successes = sum(1 for _, fy in offspring if fy < f_best)
                # Actually, standard 1/5 rule compares against previous best.
                # Here we just use a simple heuristic or count improvement over x_best before update.
                # Let's stick to a simpler adaptation: if we improved, slightly increase sigma, else decrease.
                # Better: Count how many offspring are better than the *old* best.
                # We didn't store old best separately in loop logic above easily, so let's just use a fixed factor based on improvement magnitude or just simple decay/growth.
                # Let's use the standard 1/5 rule logic properly:
                # Success if any offspring is better than the parent.
                # If success rate > 0.2, increase sigma, else decrease.
                # For (1+lambda), it's slightly different. Let's use a simpler adaptive scheme:
                # If improved, sigma *= 1.05, else sigma *= 0.95.
                # This is robust and simple.
                sigma = min(sigma * 1.05, sigma_max)
            else:
                no_improve += 1
                sigma = max(sigma * 0.95, sigma_min)
                
            if no_improve >= stall_threshold:
                break
                
        return x_best, f_best, evals

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 325  # Safety margin
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Phase 1: Random Initialization
        # Use a small portion of the budget for global exploration
        n_init = min(20, budget // 10)
        best_x = None
        best_f = float('inf')
        init_points = []
        
        for _ in range(n_init):
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            init_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        evals = n_init
        
        # Phase 2: Local Refinement with Restarts
        # We will try to refine the best point found.
        # If it stalls, we restart from the best point with a smaller sigma,
        # or try another good point from initialization.
        
        # Sort init points by fitness
        init_points.sort(key=lambda t: t[1])
        
        current_sigma = 1.0
        max_restarts = 10
        restarts_used = 0
        
        # Track best overall
        global_best_x = best_x.copy()
        global_best_f = best_f
        
        while evals < budget and time.time() < deadline and restarts_used < max_restarts:
            # Allocate remaining budget
            remaining = budget - evals
            if remaining <= 0:
                break
            
            # Choose starting point
            # Use the best point found so far
            x0 = global_best_x.copy()
            
            # Run ES
            new_best_x, new_best_f, evals_used = self._run_1plus_lambda_es(
                func, x0, current_sigma, remaining, rng, deadline
            )
            
            evals += evals_used
            
            if new_best_f < global_best_f:
                global_best_x = new_best_x
                global_best_f = new_best_f
                # If we improved, we can try to continue with a slightly smaller sigma
                # or restart to see if we can squeeze more.
                # Since (1+lambda) adapts sigma, if it stalled, it's likely local min or noise.
                # Restarting with a smaller sigma might help if the landscape is very flat/ill-conditioned.
                current_sigma *= 0.5
            else:
                # No improvement. Try a different starting point from init list?
                # Or just reduce sigma significantly and restart from best?
                # For unimodal functions, reducing sigma is good.
                # For multimodal, we'd want to jump. f11 is unimodal.
                current_sigma *= 0.5
                if current_sigma < 1e-4:
                    break
                    
            restarts_used += 1
            
            # Check if we've converged
            if global_best_f < -50: # Heuristic for f11 optimum range
                break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
