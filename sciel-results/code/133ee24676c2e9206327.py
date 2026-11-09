"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Random restart (1+1)-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls = 0
        
        def check_deadline():
            return calls >= self.budget or time.time() >= deadline
        
        # Random restart (1+1)-ES
        max_trials = 15
        calls_per_trial = self.budget // max_trials
        
        for trial in range(max_trials):
            if check_deadline():
                break
            
            # Random start
            x = rng.uniform(-4.5, 4.5, self.dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.0
            # Local refinement with (1+1)-ES
            for step in range(calls_per_trial - 1):
                if check_deadline():
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.5
                else:
                    sigma *= 0.87
        
        # Final refinement from best_x
        if best_x is not None and calls < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            while calls < self.budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                    if sigma < 1e-8:
                        break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
