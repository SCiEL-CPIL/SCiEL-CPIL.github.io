"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start with adaptive local search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Coarse random sampling to find promising regions
        n_coarse = min(50, budget // 6)
        coarse_x = rng.uniform(-5, 5, (n_coarse, dim))
        coarse_f = np.array([func(x) for x in coarse_x])
        
        # Select top k points as starting points for local refinement
        k = min(5, n_coarse)
        top_idx = np.argsort(coarse_f)[:k]
        
        best_x = coarse_x[top_idx[0]].copy()
        best_f = coarse_f[top_idx[0]]
        
        calls_used = n_coarse
        time_start = time.time()
        deadline = time_start + 280  # Safety margin before 300s limit
        
        # Phase 2: Local refinement from each top starting point
        # Use a (1+1)-ES style local search with adaptive step size
        for start_idx in top_idx:
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
                
            x = coarse_x[start_idx].copy()
            fx = coarse_f[start_idx]
            sigma = 1.0
            
            # Local search budget: distribute remaining budget among k starts
            local_budget = min(budget - calls_used, max(10, (budget - calls_used) // (k - start_idx)))
            
            for i in range(local_budget):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
        
        return best_x, best_f
# EVOLVE-BLOCK-END
