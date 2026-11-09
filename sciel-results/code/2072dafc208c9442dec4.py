"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        
        best_x = None
        best_fx = np.inf
        
        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

        # Phase 1: Exploration
        # Evaluate a set of diverse random points to find promising basins.
        # Weierstrass is multimodal, so starting points matter.
        n_explore = min(20, budget // 5)
        if n_explore < 5:
            n_explore = budget
            n_exploit = 1
        else:
            n_exploit = 5

        explore_points = rng.uniform(-5, 5, (n_explore, n))
        explore_values = np.array([func(p) for p in explore_points])
        
        # Update best with all explore points
        for i in range(n_explore):
            update_best(explore_points[i], explore_values[i])

        # Select top n_exploit points for exploitation
        # Sort by value, take the best ones
        sorted_indices = np.argsort(explore_values)
        best_seeds = explore_points[sorted_indices[:n_exploit]]
        best_seed_values = explore_values[sorted_indices[:n_exploit]]

        # Phase 2: Exploitation
        # Allocate remaining budget to the best seeds
        remaining_budget = budget - n_explore
        if remaining_budget <= 0:
            return best_x, best_fx
        
        budget_per_seed = remaining_budget // n_exploit
        # If there's a remainder, give it to the best seed
        extra = remaining_budget % n_exploit

        for i in range(n_exploit):
            if time.time() > deadline:
                break
                
            x = best_seeds[i].copy()
            fx = best_seed_values[i]
            
            # Adaptive step size (1/5 rule)
            sigma = 0.5
            successes = 0
            window = 15
            
            alloc = budget_per_seed + (extra if i == 0 else 0)
            
            for step in range(alloc):
                if time.time() > deadline:
                    return best_x, best_fx
                    
                # Generate candidate
                y = x + sigma * rng.standard_normal(n)
                # Clip to box
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                
                # Update global best
                update_best(y, fy)
                
                # Update local best for this run
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    sigma *= 1.5
                else:
                    sigma *= 0.85
                
                # Adapt sigma based on success rate in last 'window' steps
                # Simple version: reset counter periodically or use a moving average
                # For simplicity and speed, we just use the multiplicative factors.
                # A more robust 1/5 rule would track successes over a window.
                # Given the short budget per seed, simple adaptation is often sufficient.
                
        return best_x, best_fx
# EVOLVE-BLOCK-END
