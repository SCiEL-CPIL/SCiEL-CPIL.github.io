"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts and deadline guard for f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def time_left():
            return deadline - time.time()

        def eval_point(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Strategy: Multi-start with a robust (1+1)-ES that adapts step size.
        # We will perform several independent restarts, each using a portion of the budget.
        # To ensure early progress, we start with a broad random search, then refine.
        
        n_starts = 15
        # Allocate budget per start, but allow dynamic reallocation if a start is very successful
        # Simple approach: each start gets roughly budget / n_starts, but we can extend if it improves.
        
        for start_idx in range(n_starts):
            if time_left() < 10.0 or calls_used >= budget:
                break

            # Initialize starting point
            # First few starts are more diverse, later ones refine around best found so far
            if start_idx == 0:
                x = rng.uniform(-5, 5, dim)
            elif start_idx == 1:
                x = np.zeros(dim)
            elif start_idx == 2:
                x = rng.uniform(-4, 4, dim)
            elif start_idx == 3:
                x = rng.uniform(-3, 3, dim)
            else:
                # Perturb the best found so far
                if best_x is not None:
                    x = best_x + rng.normal(0, 2.0, dim)
                else:
                    x = rng.uniform(-5, 5, dim)
            
            x = np.clip(x, -5, 5)
            
            fx = eval_point(x)
            if fx is None:
                break

            sigma = 1.5
            no_improve = 0
            max_no_improve = 20 # Restart local search if stuck
            
            # Local search loop
            while calls_used < budget and time_left() > 5.0:
                # Adaptive step size
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = eval_point(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    no_improve = 0
                else:
                    sigma *= 0.7
                    no_improve += 1
                
                if no_improve > max_no_improve:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
