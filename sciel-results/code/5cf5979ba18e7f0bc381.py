"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step for multimodal Weierstrass."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 290.0  # safety margin before 300s kill
        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start exploration with 5 candidate locations
        n_starts = 5
        per_start_budget = max(50, (self.budget - n_starts) // n_starts)
        
        for s in range(n_starts):
            if time.time() >= deadline:
                break
            if calls_made >= self.budget:
                break
            
            # Generate diverse starting points
            if s == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-4.5, 4.5, self.dim)
            
            fx = evaluate(x)
            sigma = 1.5
            
            # (1+1)-ES with adaptive step
            no_improve = 0
            for _ in range(per_start_budget):
                if time.time() >= deadline:
                    break
                if calls_made >= self.budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.08  # faster increase
                    no_improve = 0
                else:
                    sigma *= 0.92  # faster decrease
                    no_improve += 1
                    
                    # If stuck for too long, restart locally
                    if no_improve > 30:
                        break

        # Phase 2: Final polish around the best point found
        if best_x is not None and calls_made < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            no_improve = 0
            polish_budget = min(50, self.budget - calls_made)
            
            for _ in range(polish_budget):
                if time.time() >= deadline:
                    break
                if calls_made >= self.budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.90
                    no_improve += 1
                    if no_improve > 20:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
