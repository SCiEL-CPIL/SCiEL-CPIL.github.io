"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with restarts and deadline guard for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def safe_func(x):
            nonlocal calls_used
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            x = np.clip(x, -5.0, 5.0)
            calls_used += 1
            return float(func(x))
        
        # Multi-restart strategy
        n_restarts = 8
        budget_per_restart = self.budget // n_restarts
        
        for r in range(n_restarts):
            if calls_used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if r == 0:
                # Start near origin (Rastrigin global min is often near 0 after rotation)
                x = np.zeros(self.dim)
            elif r == 1:
                # Small random perturbation from origin
                x = rng.uniform(-1, 1, self.dim)
            else:
                # Random points with decreasing spread
                spread = 5.0 * (0.7 ** r)
                x = rng.uniform(-spread, spread, self.dim)
            
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.0
            sigma_min = 1e-6
            sigma_max = 5.0
            no_improve = 0
            max_no_improve = 30
            
            while calls_used < self.budget:
                if time.time() > deadline:
                    break
                if no_improve > max_no_improve:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, sigma_max)
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.8, sigma_min)
                    no_improve += 1
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = safe_func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
