"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start random search with local (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_used = 0

        # We'll do multiple restarts. Each restart:
        # 1. Pick a random starting point (biased toward center)
        # 2. Run a short (1+1)-ES for a few steps
        # 3. Keep the best overall

        # Allocate budget across restarts. 
        # Let's say we do up to 20 restarts, each getting ~25 calls.
        # But we adapt: if we're making good progress, fewer restarts.
        # Simple approach: fixed number of restarts, each with a fixed local budget.
        
        n_restarts = 20
        local_budget = self.budget // n_restarts  # 25 calls per restart

        for r in range(n_restarts):
            # Check time
            if time.time() > deadline:
                break
            if calls_used >= self.budget:
                break

            # Initialize: mix of center-biased and random
            if r == 0:
                x = np.zeros(self.dim)
            else:
                # Gaussian around origin, clipped to box
                x = np.clip(rng.normal(0, 1.5, self.dim), -5, 5)
            
            fx = func(x)
            calls_used += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Local (1+1)-ES refinement
            sigma = 1.0
            for step in range(local_budget - 1):
                if time.time() > deadline:
                    break
                if calls_used >= self.budget:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    # If sigma gets too small, break early and restart
                    if sigma < 0.01:
                        break
            
            # If we broke early due to small sigma, we'll restart anyway
            # which is good for exploration

        return best_x, best_f
# EVOLVE-BLOCK-END
