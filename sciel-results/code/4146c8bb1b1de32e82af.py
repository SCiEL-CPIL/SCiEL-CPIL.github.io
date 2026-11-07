"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, adaptive restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285  # 300s limit, 15s safety margin
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        dim = self.dim
        total_budget = self.budget

        # Strategy: Use a small number of diversified starts, then run (1+1)-ES
        # from the best one. This helps with hidden shifts/rotations.
        
        # Phase 1: Diversified initialization
        # We allocate a small fraction of the budget to sampling.
        # 10 samples is cheap (2% of 500 budget) and helps avoid bad local minima.
        n_starts = 10
        candidates = []
        
        # Include center and various scales
        candidates.append(np.zeros(dim))
        for i in range(n_starts - 1):
            # Mix of different scales to cover the [-5, 5] box
            if i < 2:
                # Tight around center
                x0 = rng.uniform(-1, 1, dim)
            elif i < 5:
                # Medium spread
                x0 = rng.uniform(-3, 3, dim)
            else:
                # Wide spread
                x0 = rng.uniform(-5, 5, dim)
            candidates.append(x0)

        for x0 in candidates:
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break
            fx = func(x0)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Phase 2: (1+1)-ES from the best starting point
        if best_x is None:
            # Fallback if something went wrong (should not happen)
            x = np.zeros(dim)
            fx = func(x)
            calls_made += 1
            best_x = x.copy()
            best_f = fx
        else:
            x = best_x.copy()
            fx = best_f

        sigma = 2.0
        # 1/5 rule parameters
        # We will use a simple adaptive scheme:
        # If success rate > 0.3, increase sigma
        # If success rate < 0.1, decrease sigma
        # Otherwise, keep sigma
        
        success_counter = 0
        total_steps = 0
        window = 10
        
        while calls_made < total_budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                success_counter += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # No improvement
                pass
                
            total_steps += 1
            
            # Adapt sigma based on success rate in the last 'window' steps
            if total_steps >= window:
                success_rate = success_counter / total_steps
                if success_rate > 0.3:
                    sigma *= 1.5
                elif success_rate < 0.1:
                    sigma *= 0.5
                else:
                    # Slight adjustment or no change
                    sigma *= 1.0 
                
                success_counter = 0
                total_steps = 0
                
                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-4, 10.0)
                
                # Restart logic: If we've been stuck for a while (low success)
                # and sigma is very small, try a restart from best_x with a larger sigma
                # This is a heuristic to escape local minima if the landscape is deceptive
                if success_rate < 0.1 and sigma < 0.1:
                    # Restart from best_x with a moderate sigma
                    x = best_x + rng.normal(0, 1.0, dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls_made += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = 2.0
                    success_counter = 0
                    total_steps = 0
                    # Check budget again
                    if calls_made >= total_budget:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
