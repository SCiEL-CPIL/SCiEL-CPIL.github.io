import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        self.eval_count = 0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Wrapper to count evaluations
        def wrapped_problem(x):
            self.eval_count += 1
            if self.eval_count > self.budget:
                # Return a large penalty to discourage further exploration in this run
                # But we should really stop. Since we use maxfev, this shouldn't happen often.
                # If it does, return large values.
                return [1e10, 1e10]
            return problem(x)

        n_starts = 10
        # Distribute budget among starts
        # Leave some buffer for safety
        budget_per_start = max(5, (self.budget - n_starts) // n_starts)
        
        rng = np.random.default_rng(42)
        
        for i in range(n_starts):
            if self.eval_count >= self.budget:
                break
            
            # Random initial point
            x0 = lo + (hi - lo) * rng.random(self.dim)
            
            # Random weight vector for scalarization
            # Use random weights to explore different parts of the Pareto front
            w = rng.random(self.n_obj) + 0.1 # Avoid zero weights
            w = w / np.sum(w) # Normalize
            
            def scalarized(x):
                f = wrapped_problem(x)
                return np.dot(w, f)
            
            # Run Nelder-Mead
            # maxfev controls the number of function evaluations
            res = minimize(
                scalarized, 
                x0, 
                method='Nelder-Mead',
                options={
                    'maxiter': 50, # Iterations
                    'maxfev': budget_per_start, # Function evaluations
                    'xatol': 1e-8,
                    'fatol': 1e-8
                }
            )
            
            # The evaluations inside minimize are already counted by wrapped_problem
            # The final point res.x is also evaluated during the process.
            
        return None
