"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = float('inf')
        calls = 0
        start_time = time.time()
        max_time = 280  # Safety margin for 300s limit
        
        def can_continue():
            return calls < self.budget and (time.time() - start_time) < max_time

        # Phase 1: Initial Exploration
        # Sample a few points to find a reasonable starting region
        n_initial = 10
        for _ in range(n_initial):
            if not can_continue():
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Iterated Local Search with Restarts
        while can_continue():
            # Determine next starting point based on current best quality
            r = rng.random()
            if best_f > 10.0:
                # Far from optimum, explore widely
                if r < 0.8:
                    x = rng.uniform(-5, 5, self.dim)
                else:
                    x = np.clip(best_x + rng.normal(0, 2.0, self.dim), -5, 5)
            elif best_f > 1.0:
                # Getting closer, mix exploration and exploitation
                if r < 0.5:
                    x = rng.uniform(-5, 5, self.dim)
                else:
                    x = np.clip(best_x + rng.normal(0, 1.0, self.dim), -5, 5)
            elif best_f > 0.1:
                # Close, exploit locally
                if r < 0.2:
                    x = rng.uniform(-5, 5, self.dim)
                else:
                    x = np.clip(best_x + rng.normal(0, 0.5, self.dim), -5, 5)
            else:
                # Very close, refine
                x = np.clip(best_x + rng.normal(0, 0.1, self.dim), -5, 5)

            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Local Search (1+1)-ES
            sigma = 0.5
            no_improve = 0
            max_local_steps = 20
            
            for _ in range(max_local_steps):
                if not can_continue():
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.7
                    no_improve += 1
                    if no_improve > 8:
                        break
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
