import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Optimizer using multi-start L-BFGS-B for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                # Return current best to signal no improvement / stop
                return best_f
            
            # Clip to box
            x = np.clip(x, -5.0, 5.0)
            
            call_count += 1
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        # 1. Generate diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        
        # Random points
        # Use different seeds for each instance if possible, or just default
        for _ in range(9):
            starts.append(np.random.uniform(-5.0, 5.0, self.dim))
            
        # 2. Evaluate starting points
        start_values = []
        for s in starts:
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break
            fs = wrapped_func(s)
            start_values.append((fs, s))
            
        if not start_values:
            # Fallback if something went wrong
            if best_x is None:
                best_x = np.zeros(self.dim)
                best_f = func(best_x)
            return best_x, best_f

        # Sort by function value
        start_values.sort(key=lambda x: x[0])
        
        # 3. Run L-BFGS-B from the best few starts
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        
        # Try top 3 starts
        for fs, s in start_values[:3]:
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break
            
            try:
                # maxfev is a soft limit in scipy, but our wrapper enforces the hard limit
                # We give it the remaining budget
                remaining = self.budget - call_count
                if remaining <= 0:
                    break
                    
                result = minimize(
                    wrapped_func, 
                    s, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 1000, 
                        'maxfev': remaining,
                        'gtol': 1e-5
                    }
                )
            except Exception:
                # If minimize fails, we still have the best_x from wrapped_func
                pass

        return best_x, best_f
