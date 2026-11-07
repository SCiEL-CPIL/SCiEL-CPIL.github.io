import numpy as np


class Optimizer:
    """Optimizer using boundary probing for linear functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # Start at the center
        x = np.zeros(d)
        fx = func(x)
        calls = 1
        
        # Determine the optimal boundary for each dimension
        # For a linear function f(x) = w^T x + c, the minimum is at a vertex.
        # We determine the sign of w_i by comparing f(x + eps e_i) and f(x - eps e_i).
        eps = 1.0
        
        for i in range(d):
            if calls >= budget:
                break
                
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            
            fp = func(x_plus)
            calls += 1
            fm = func(x_minus)
            calls += 1
            
            if fm < fp:
                # w_i is positive, so we want x_i to be as small as possible (-5)
                x[i] = -5.0
            else:
                # w_i is negative (or zero), so we want x_i to be as large as possible (5)
                # If w_i is 0, it doesn't matter, but 5 is fine.
                x[i] = 5.0
                
            fx = func(x)
            calls += 1
            
            if calls >= budget:
                break
                
        # Use remaining budget for local refinement if needed, 
        # though for linear functions the vertex is optimal.
        # We can do a simple (1+1)-ES polish to be safe against any non-linearity.
        sigma = 0.1
        for _ in range(budget - calls):
            if calls >= budget:
                break
            y = np.clip(x + sigma * np.random.default_rng(calls).standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
                if sigma < 1e-6:
                    break
                    
        return x, fx
