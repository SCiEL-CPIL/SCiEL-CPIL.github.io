"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multiple random restarts and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Multiple random restarts with short (1+1)-ES runs
        # This gives early progress by exploring different regions
        num_restarts = 5
        evals_per_restart = self.budget // num_restarts
        
        for restart in range(num_restarts):
            # Check time
            if time.time() > deadline:
                break
            
            # Start from a random point, biased toward center
            if restart == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-3, 3, self.dim)
            
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.5
            local_best_x = x.copy()
            local_best_f = fx
            
            # Short (1+1)-ES run
            for _ in range(evals_per_restart - 1):
                # Check time
                if time.time() > deadline:
                    break
                
                y = np.clip(local_best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                
                if fy < local_best_f:
                    local_best_x, local_best_f = y, fy
                    sigma *= 1.3
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                else:
                    sigma *= 0.85
            
            # If we found a better solution, continue from there
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()
        
        # Phase 2: Continue with the best solution using remaining budget
        if best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Smaller step for fine-tuning
            
            remaining_budget = self.budget - (num_restarts * evals_per_restart)
            for _ in range(remaining_budget):
                # Check time
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
        
        return best_x, best_f
# EVOLVE-BLOCK-END
