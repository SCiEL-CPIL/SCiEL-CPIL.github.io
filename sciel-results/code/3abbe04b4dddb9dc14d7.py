"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Track global best
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Number of restarts - adaptive: more restarts early, fewer later
        # Use a mix: some full-budget runs, some shorter
        num_restarts = 8
        
        for r in range(num_restarts):
            # Check time
            if time.time() > deadline:
                break
            
            # Determine budget for this restart
            if r == 0:
                # First restart: full exploration
                restart_budget = min(self.budget - calls_made, 200)
            elif r == 1:
                restart_budget = min(self.budget - calls_made, 150)
            elif r == 2:
                restart_budget = min(self.budget - calls_made, 100)
            else:
                # Later restarts: smaller budgets, seeded near best
                restart_budget = min(self.budget - calls_made, 50)
            
            if restart_budget <= 0:
                break
            
            # Initialize x
            if r == 0:
                x = rng.uniform(-5, 5, self.dim)
                sigma = 2.0
            else:
                # Seed near best point with some perturbation
                if best_x is not None:
                    # Perturbation scales with sigma-like quantity
                    perturb = 1.0 * np.sqrt(self.dim)
                    x = np.clip(best_x + perturb * rng.standard_normal(self.dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, self.dim)
                sigma = 1.0
            
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Run (1+1)-ES for this restart
            for i in range(restart_budget - 1):
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                
                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 10.0)
            
            if calls_made >= self.budget:
                break
        
        if best_x is None:
            # Fallback: should not happen, but safe
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
