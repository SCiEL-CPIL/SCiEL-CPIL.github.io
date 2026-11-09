"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, initial sampling, and robust stall handling."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, aim for 270s to be safe
        deadline = start_time + 270 
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial sampling to find a good starting point
        # We use a few random points to avoid starting in a bad local region
        best_x = None
        best_fx = np.inf
        n_init = min(15, budget)
        
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)

        x = best_x.copy()
        fx = best_fx
        calls_used = n_init if best_x is not None else 1

        # (1+1)-ES parameters
        # Start with a moderate step size
        sigma = 1.0
        stall_count = 0
        max_stall = 30
        # Success rule: 1/5 -> multiply by 1.3 on success, 0.9 on failure
        # This keeps the success rate around 20%, which is optimal for (1+1)-ES
        
        # Main loop: (1+1)-ES
        while calls_used < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                # Success
                x, fx = y, fy
                sigma *= 1.3
                stall_count = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Failure
                sigma *= 0.9
                stall_count += 1

            # Stall detection: if we haven't improved in a while, try to escape
            if stall_count >= max_stall:
                stall_count = 0
                # Reduce sigma to allow finer search, but also try a larger jump to escape local minima
                # We reset sigma to a value that allows exploration
                sigma = max(sigma * 0.5, 0.1)
                
                # Try a few local perturbations around the best point found so far
                # This helps if we are stuck in a local minimum that is not the global one
                for _ in range(5):
                    if time.time() > deadline or calls_used >= budget:
                        break
                    # Larger step to potentially jump out of the local basin
                    jump_sigma = max(sigma * 2.0, 0.5)
                    y = np.clip(best_x + jump_sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_used += 1
                    
                    if fy < fx:
                        x, fx = y, fy
                        # If we found a better point, reset sigma to moderate
                        sigma = max(sigma, 0.5) 
                        break
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()
                        # If we found a new global best, reset current x to this
                        x = best_x.copy()
                        fx = best_fx

            # Early exit if sigma is extremely small and we are stuck
            if sigma < 1e-6 and stall_count > 10:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
