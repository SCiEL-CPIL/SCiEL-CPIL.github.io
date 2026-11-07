"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with multi-start init and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        
        calls_made = 0
        
        # Phase 1: Multi-start initialization
        n_starts = 10
        best_fx = np.inf
        best_x = np.zeros(self.dim)
        
        for _ in range(n_starts):
            if time.time() > deadline or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Phase 2: (1+lambda)-ES refinement
        x = best_x.copy()
        fx = best_fx
        
        sigma = 1.0
        lambda_ = 8
        c_sigma = 0.8
        target_success = 0.2
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            improvements = 0
            best_y = None
            best_fy = fx
            
            for j in range(lambda_):
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break
                    
                step = sigma * rng.standard_normal(self.dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < best_fy:
                    best_fy = fy
                    best_y = y
                    improvements += 1
            
            # Adapt step size
            actual_success = improvements / lambda_
            if actual_success > target_success:
                sigma *= (1 + c_sigma * (actual_success - target_success))
            else:
                sigma *= (1 - c_sigma * (target_success - actual_success))
            
            sigma = np.clip(sigma, 0.01, 10.0)
            
            # Accept best offspring
            if best_y is not None:
                x = best_y
                fx = best_fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
