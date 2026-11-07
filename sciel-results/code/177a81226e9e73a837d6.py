"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned) on [-5,5]^5.
    
    Strategy: Robust Multi-start Adaptive (1+1)-ES.
    - Uses a population of random starts.
    - Each start runs an adaptive (1+1)-ES with stagnation detection.
    - Stagnation triggers a restart of the local search or a new global start.
    - Final phase polishes the global best.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Configuration
        n_restarts = 12
        reserve_budget = 40
        search_budget = budget - reserve_budget
        budget_per_restart = max(10, search_budget // n_restarts)
        
        # Phase 1: Multi-start exploration with adaptive local search
        for r in range(n_restarts):
            if calls_used >= budget - reserve_budget:
                break
                
            # Random start point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Local Adaptive (1+1)-ES
            sigma = 1.0
            local_budget = min(budget_per_restart, budget - calls_used - reserve_budget)
            no_improve_count = 0
            max_no_improve = 15  # Stagnation threshold
            
            for _ in range(local_budget):
                if calls_used >= budget - reserve_budget:
                    break
                if no_improve_count >= max_no_improve:
                    break  # Stagnation, move to next restart
                    
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    no_improve_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve_count += 1
                    
                    # If sigma becomes too small, increase it to escape
                    if sigma < 0.05:
                        sigma = 0.5
            
        # Phase 2: Final Polish
        if best_x is not None and calls_used < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = budget - calls_used
            
            for _ in range(remaining):
                if calls_used >= budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    if sigma < 0.01:
                        sigma = 0.1

        return best_x, best_f
# EVOLVE-BLOCK-END
