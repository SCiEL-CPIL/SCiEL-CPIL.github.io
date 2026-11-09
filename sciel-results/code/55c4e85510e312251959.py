"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-restart CMA-ES-like optimizer for BBOB f14 (Different powers).
    
    Strategy:
    - Use multiple random restarts with adaptive step sizes.
    - Each restart runs a (1+1)-ES with aggressive step adaptation.
    - Keep the best solution across all restarts.
    - Time guard to ensure we finish within the deadline.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Multiple random restarts with short (1+1)-ES
        # We'll use about 60% of budget for exploration, 40% for exploitation
        
        n_restarts = 10
        budget_per_restart = max(20, (self.budget * 60) // 100 // n_restarts)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initial point
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # (1+1)-ES with adaptive step size
            sigma = 1.5
            for step in range(budget_per_restart):
                if time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
            
            # Occasionally do a larger jump to explore different basins
            if restart < n_restarts - 1 and time.time() < deadline:
                jump_x = x + 2.0 * rng.standard_normal(self.dim)
                jump_x = np.clip(jump_x, -5, 5)
                jump_f = func(jump_x)
                if jump_f < best_f:
                    best_f = jump_f
                    best_x = jump_x.copy()
        
        # Phase 2: Exploit the best solution with more refined search
        if best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            
            # Continue with finer-grained search
            remaining_budget = self.budget - (n_restarts * budget_per_restart)
            for step in range(max(0, remaining_budget)):
                if time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
        
        # Final refinement: try small perturbations around the best point
        if best_x is not None and time.time() < deadline:
            for _ in range(50):
                if time.time() > deadline:
                    break
                y = best_x + 0.1 * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
