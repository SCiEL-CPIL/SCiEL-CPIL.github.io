"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with a final polish phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: return before 330s limit
        timeout = 290.0
        
        x_best = np.zeros(self.dim)
        f_best = float('inf')
        
        # Phase 1: Exploration with 3 restarts
        # 100 evaluations per restart
        n_restarts = 3
        evals_per_restart = 100
        
        # Initial points: 0, and two random points
        # Using fixed seeds for the internal RNG ensures consistent behavior
        # across instances, but the instance transformation makes the
        # effective starting locations different.
        initial_points = [
            np.zeros(self.dim),
            np.random.default_rng(1).uniform(-4, 4, self.dim),
            np.random.default_rng(2).uniform(-4, 4, self.dim)
        ]
        
        for r in range(n_restarts):
            if time.time() - start_time > timeout:
                break
                
            x = initial_points[r].copy()
            fx = func(x)
            
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
                
            sigma = 1.0
            rng = np.random.default_rng(100 + r)
            
            for _ in range(evals_per_restart):
                if time.time() - start_time > timeout:
                    break
                    
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, self.bounds[0], self.bounds[1])
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.3
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                else:
                    sigma *= 0.7

        # Phase 2: Exploitation / Polish
        # Use remaining budget (200) to refine x_best
        remaining_budget = self.budget - (n_restarts * evals_per_restart)
        if remaining_budget > 0:
            if time.time() - start_time > timeout:
                return x_best, f_best
                
            x = x_best.copy()
            fx = f_best
            sigma = 0.1 # Small step size for polishing
            rng = np.random.default_rng(999)
            
            for _ in range(remaining_budget):
                if time.time() - start_time > timeout:
                    break
                    
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, self.bounds[0], self.bounds[1])
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    if fx < f_best:
                        f_best = fx
                        x_best = x.copy()
                else:
                    sigma *= 0.8

        return x_best, f_best
# EVOLVE-BLOCK-END
