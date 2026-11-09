"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and restarts for Rastrigin rotated."""

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
        
        # Phase 1: Broad exploration with multiple random starts
        n_explore = 10
        for i in range(n_explore):
            if calls_used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = safe_func(best_x)
        
        # Phase 2: Local refinement from best point using (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        sigma_min = 1e-8
        sigma_max = 3.0
        no_improve = 0
        max_no_improve = 40
        
        while calls_used < self.budget:
            if time.time() > deadline:
                break
            if no_improve > max_no_improve:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = safe_func(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.1, sigma_max)
                no_improve = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma = max(sigma * 0.9, sigma_min)
                no_improve += 1
        
        # Phase 3: If still have budget, try a few more restarts from diverse points
        remaining_budget = self.budget - calls_used
        if remaining_budget > 50:
            n_restarts = min(3, remaining_budget // 50)
            for r in range(n_restarts):
                if calls_used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Initialize near best_x but with some perturbation
                x = np.clip(best_x + rng.uniform(-2, 2, self.dim), -5, 5)
                fx = safe_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                sigma = 0.3
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
                        sigma = min(sigma * 1.15, sigma_max)
                        no_improve = 0
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        sigma = max(sigma * 0.85, sigma_min)
                        no_improve += 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
