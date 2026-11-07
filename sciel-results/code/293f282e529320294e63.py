"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size and deadline guard for f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        best_x = None
        best_f = np.inf
        calls = 0
        budget = self.budget
        dim = self.dim

        def time_left():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-restart (1+1)-ES
        # We divide the budget among several restarts.
        # For f21 (Gallagher 101 peaks), multiple restarts help escape local minima.
        # Let's try 5 restarts with 100 calls each, or adapt based on progress.
        
        # Actually, let's use a simpler strategy: 
        # Run (1+1)-ES for a chunk of budget, then restart from best point with new noise.
        # Or: run several independent (1+1)-ES instances in parallel (sequentially) with different seeds.
        
        # Strategy: 5 restarts, 100 calls each.
        n_restarts = 5
        calls_per_restart = budget // n_restarts  # 100 calls each
        
        for r in range(n_restarts):
            if calls >= budget or time_left() < 5.0:
                break
            
            # Initialize: for first restart, random; for subsequent, near best with perturbation
            if r == 0:
                rng = np.random.default_rng(42 + r)
                x = rng.uniform(-5, 5, dim)
            else:
                if best_x is not None:
                    rng = np.random.default_rng(42 + r)
                    x = best_x + rng.normal(0, 1.0, dim)
                    x = np.clip(x, -5, 5)
                else:
                    rng = np.random.default_rng(42 + r)
                    x = rng.uniform(-5, 5, dim)
            
            fx = evaluate(x)
            sigma = 2.0
            max_sigma = 4.0
            min_sigma = 1e-4
            
            for _ in range(calls_per_restart):
                if calls >= budget or time_left() < 2.0:
                    break
                
                y = x + sigma * np.random.default_rng().standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, max_sigma)
                else:
                    sigma = max(sigma * 0.85, min_sigma)
        
        # Final polish: use remaining budget for a fine-grained local search around best_x
        if best_x is not None and calls < budget and time_left() > 1.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = budget - calls
            for _ in range(remaining):
                if time_left() < 0.5:
                    break
                y = x + sigma * np.random.default_rng().standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma = max(sigma * 0.95, 1e-5)
                else:
                    sigma = max(sigma * 0.9, 1e-5)

        if best_x is None:
            # Should not happen, but safeguard
            x = np.zeros(dim)
            best_x = x
            best_f = func(x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
