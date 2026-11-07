import numpy as np
import time
from scipy.optimize import minimize

class BudgetExceeded(Exception):
    pass

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget
        
        # Track state
        calls = 0
        best_x = None
        best_fx = np.inf
        
        # Wrapper to count calls and enforce budget
        def wrapped_func(x):
            nonlocal calls, best_x, best_fx
            if time.time() > deadline:
                raise BudgetExceeded("Time limit exceeded")
            
            # Clip to domain
            x_clipped = np.clip(x, -5.0, 5.0)
            
            # Call the actual function
            fx = func(x_clipped)
            calls += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
                
            if calls >= budget:
                raise BudgetExceeded("Budget exceeded")
                
            return fx

        # Initial guess
        # Bent Cigar minimum is shifted. A random start is safer than 0 if shift is large.
        # But L-BFGS-B is good at navigating. Let's try a few random starts or just one good one.
        # Given the budget is 500, we can afford a small initial exploration.
        # However, L-BFGS-B is a local optimizer. If we start far from the min, it might converge to a bad local min?
        # Bent Cigar is convex, so there is only one global min. L-BFGS-B will converge to it from anywhere.
        
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-4.0, 4.0, dim)
        
        # Initialize best with x0
        wrapped_func(x0)
        
        try:
            # Use L-BFGS-B
            # options: maxiter, ftol, gtol
            # We want to use as many calls as possible, so set tolerances loose.
            res = minimize(
                wrapped_func, 
                x0, 
                method='L-BFGS-B',
                bounds=[(-5.0, 5.0)] * dim,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
        except BudgetExceeded:
            pass
        except Exception:
            # Catch any other errors to ensure we return something
            pass

        # If we didn't update best_x (shouldn't happen as we called wrapped_func once), use x0
        if best_x is None:
            best_x = x0
            best_fx = func(np.clip(x0, -5.0, 5.0))

        return best_x, best_fx
