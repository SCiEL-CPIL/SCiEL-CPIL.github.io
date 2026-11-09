"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Portfolio optimizer for BBOB f14 (Different powers).
    
    Strategy:
    1. Brief exploration phase with multiple short (1+1)-ES restarts.
    2. Long exploitation phase on the best individual with adaptive step size.
    3. Time-aware: stops before wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f
        
        # Phase 1: Exploration with multiple short restarts
        n_restarts = 15
        calls_per_restart = 25  # total 375 calls for exploration
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break
            
            # Start near origin with some randomness
            if r == 0:
                x = rng.uniform(-1, 1, self.dim)
            else:
                x = rng.uniform(-5, 5, self.dim)
            
            fx = func(x)
            calls += 1
            try_update(x, fx)
            
            sigma = 1.5
            for _ in range(calls_per_restart):
                if time.time() > deadline:
                    break
                if calls >= self.budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                try_update(y, fy)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9
                
                if sigma < 1e-6:
                    break
        
        # Phase 2: Exploitation on the best individual found
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            
            remaining = self.budget - calls
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if calls >= self.budget:
                    break
                
                # Occasionally try a larger jump to escape local minima
                if rng.random() < 0.05:
                    jump_sigma = 2.0
                else:
                    jump_sigma = sigma
                
                y = np.clip(x + jump_sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                try_update(y, fy)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.92
                
                # Adaptive: if stuck, try increasing sigma
                if sigma < 0.01:
                    sigma = 0.1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
