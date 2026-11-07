"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Hybrid optimizer combining:
    1. A coarse random search to get a decent starting point.
    2. Nelder-Mead simplex method for local exploitation.
    3. A final polish with a (1+1)-ES or similar for fine-tuning.
    
    This approach leverages the low dimensionality (d=5) and limited budget (500).
    Random search helps escape local minima or find the right basin of attraction,
    while Nelder-Mead efficiently converges to the local minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Budget management
        calls = 0
        max_calls = self.budget

        def safe_call(x):
            nonlocal calls
            if calls >= max_calls:
                return np.inf
            calls += 1
            return func(x)

        # Phase 1: Random Search (use ~20% of budget)
        n_random = max(10, self.budget // 5)
        
        best_x = None
        best_f = np.inf
        
        # Use a fixed seed for reproducibility in case it helps, or random?
        # The task says "hidden seeded shift", so the function is deterministic per instance.
        # We should use a random seed for our search to avoid bad luck, but 
        # the evaluator might run multiple times? No, "evaluator creates Optimizer... and calls it once".
        # So we just need a good search strategy.
        
        rng = np.random.default_rng()
        
        # Initial random points
        for _ in range(n_random):
            x = rng.uniform(-5, 5, self.dim)
            f = safe_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if calls >= max_calls:
            return best_x, best_f

        # Phase 2: Nelder-Mead from the best point found
        # Nelder-Mead is derivative-free and works well for smooth, low-dim functions.
        # Schaffers F7 is smooth.
        
        # We can run Nelder-Mead with a limited number of function evaluations.
        # scipy.optimize.minimize doesn't have a direct "maxiter" that counts function calls exactly,
        # but maxiter in Nelder-Mead usually corresponds to iterations of the simplex update.
        # Each iteration might use 1 or more function calls.
        # To be safe, we can wrap the function to count calls.
        
        remaining_budget = max_calls - calls
        if remaining_budget <= 0:
            return best_x, best_f

        # Set up Nelder-Mead
        # options: maxiter, xatol, fatol, adaptive
        # Let's allocate about 60% of remaining budget to Nelder-Mead iterations?
        # Actually, Nelder-Mead in d=5 dimensions might need ~50-100 iterations to converge.
        # Each iteration uses ~d+1 = 6 function calls worst case, but often less.
        # Let's try maxiter=100.
        
        try:
            res = minimize(
                safe_call, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 150, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass # If Nelder-Mead fails, we keep the best from random search

        if calls >= max_calls:
            return best_x, best_f

        # Phase 3: Polish with a small (1+1)-ES or local perturbation
        # Use the remaining budget for a fine-grained search around best_x.
        remaining_budget = max_calls - calls
        if remaining_budget > 0:
            sigma = 0.1
            x = best_x.copy()
            fx = best_f
            
            for _ in range(remaining_budget):
                # Generate a perturbation
                delta = sigma * rng.standard_normal(self.dim)
                y = np.clip(x + delta, -5, 5)
                fy = safe_call(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    # Increase step size slightly if we're making progress?
                    # Or keep it constant for fine polish?
                    # Let's decay it slowly to ensure convergence.
                    sigma *= 0.95
                else:
                    sigma *= 0.8
                    if sigma < 1e-4:
                        break

            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f

# EVOLVE-BLOCK-END
