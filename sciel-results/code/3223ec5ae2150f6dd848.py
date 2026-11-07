"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start Nelder-Mead with coarse restarts and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(42)
        
        # State tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0
        dim = self.dim
        
        # Wrapper to count calls and enforce hard limit
        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return float('inf')
            calls_made += 1
            return func(x)

        # Phase 1: Coarse Random Restart Search
        # Strategy: Use small budgets per restart to cover the space.
        # Bent Cigar has a narrow valley. Random restarts help find the valley direction.
        # We allocate ~60% of the budget to this phase.
        # dim=5, budget=500. 
        # Nelder-Mead in 5D is ~5-10 calls per step. 
        # Let's try ~20-30 restarts with ~10-15 calls each.
        
        coarse_budget_per_restart = 12
        max_coarse_restarts = 25 # 25 * 12 = 300 calls
        
        for i in range(max_coarse_restarts):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
            
            # Random start in [-5, 5]^d
            x0 = rng.uniform(-5, 5, dim)
            
            # Nelder-Mead with tight maxiter to act as a local coarse search
            # Initial simplex size 0.5 is small enough for local refinement but 
            # we are restarting, so it explores locally around x0.
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': coarse_budget_per_restart,
                        'xatol': 1e-4, # Don't refine too much here
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Phase 2: Local Refinement
        # Use the remaining budget to refine the best point found.
        # This is where we get the precision for AOCC.
        if best_x is None:
            # Fallback if no valid result from Phase 1 (shouldn't happen)
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped_func(best_x)

        if calls_made < self.budget:
            remaining_budget = self.budget - calls_made
            # Nelder-Mead for refinement
            # We can afford more iterations now.
            # maxiter should be capped to not exceed budget.
            # NM in 5D uses ~6 calls/iter. 
            # Let's set maxiter to something reasonable, but rely on budget check.
            # Actually, scipy minimize doesn't take a "budget" directly, 
            # but our wrapped_func returns inf if over budget.
            # To be safe, we set maxiter to a high number and let the 
            # wrapped_func or time deadline stop it.
            
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': remaining_budget, # Upper bound on iters
                        'xatol': 1e-8, # High precision
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
