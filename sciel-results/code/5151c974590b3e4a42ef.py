import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        self.eval_count = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        # Wrapped function to count evaluations and update best
        def wrapped_func(x):
            self.eval_count += 1
            # Deadline check
            if time.time() - self.start_time > 280: # 20s safety margin
                return 1e100
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Generate initial candidates
        # 1. Origin
        candidates = [np.zeros(self.dim)]
        
        # 2. Random points
        rng = np.random.default_rng(42)
        for _ in range(4):
            candidates.append(rng.uniform(-5, 5, self.dim))
            
        # Evaluate candidates
        for c in candidates:
            if self.eval_count >= self.budget:
                break
            val = wrapped_func(c)

        # Sort candidates by their initial value
        # We need to store the values to sort
        cand_vals = []
        for c in candidates:
             # We already evaluated them, but we didn't store the specific value for sorting easily 
             # Let's re-evaluate or store. Storing is better.
             pass
             
        # Let's restructure to store (val, x)
        init_results = []
        for c in candidates:
            if self.eval_count >= self.budget:
                break
            val = wrapped_func(c)
            init_results.append((val, c))
            
        init_results.sort(key=lambda x: x[0])

        # Run Nelder-Mead from best candidates
        # We have a budget of ~500. We used ~5. Remaining ~495.
        # Allocate budget to NM runs.
        # If we have 5 candidates, we can give each ~90 evals.
        
        for val, x0 in init_results:
            if self.eval_count >= self.budget:
                break
            if time.time() - self.start_time > 280:
                break
                
            # Run Nelder-Mead
            # maxiter in NM is not exactly eval count, but it's a good proxy.
            # NM usually does dim+1 evals per iteration.
            # Let's limit iterations to ensure we don't overshoot.
            # Budget remaining: self.budget - self.eval_count
            # Max iterations: (remaining_budget) // (dim + 1)
            
            remaining = self.budget - self.eval_count
            if remaining < 10:
                break
                
            maxiter = max(10, remaining // (self.dim + 1))
            
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                continue
                
            # Update best if NM found something better
            # wrapped_func already updates self.best_x/f

        return self.best_x, self.best_f
