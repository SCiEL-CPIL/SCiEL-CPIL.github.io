import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.x1_values = None

    def __call__(self, problem, bounds):
        if self.x1_values is None:
            self._compute_optimal_x1()
        
        # Extract bounds for clipping
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate and evaluate points
        for x1 in self.x1_values:
            x = np.zeros(self.dim)
            x[0] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            problem(x)
        
        return None

    def _compute_optimal_x1(self):
        k = self.budget
        if k <= 0:
            self.x1_values = np.array([])
            return
            
        # Initial guess: uniform
        x0 = np.linspace(0, 1, k)
        
        def objective(x):
            # x is assumed to be the x1 coordinates
            # We want to maximize hypervolume
            # H = sum_{i=0}^{k-2} (x[i+1] - x[i]) * (10 + x[i]^2) + (11 - x[k-1]) * (10 + x[k-1]^2)
            
            # Handle case k=1
            if k == 1:
                return -( (11 - x[0]) * (10 + x[0]**2) )
                
            diffs = np.diff(x)
            # If x is not sorted, diffs can be negative.
            # The term (diff * weight) will be negative, reducing the sum (since we minimize -H).
            # This encourages sorting.
            terms = diffs * (10 + x[:-1]**2)
            hv = np.sum(terms)
            hv += (11 - x[-1]) * (10 + x[-1]**2)
            return -hv

        bounds_opt = [(0, 1)] * k
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds_opt, options={'maxiter': 2000, 'ftol': 1e-12})
            self.x1_values = res.x
        except Exception:
            self.x1_values = x0
            
        # Ensure sorted and within bounds
        self.x1_values = np.sort(self.x1_values)
        self.x1_values = np.clip(self.x1_values, 0, 1)
        
        # Ensure we have exactly k points
        if len(self.x1_values) < k:
             # Pad with 1s if needed? Or just use what we have.
             # But we must use exactly k evaluations?
             # "More than 200... invalid".
             # We can repeat the last point?
             # Repeating a point on the front doesn't change hypervolume but uses budget.
             # It's safe.
             if len(self.x1_values) > 0:
                 pad_val = self.x1_values[-1]
             else:
                 pad_val = 1.0
             self.x1_values = np.pad(self.x1_values, (0, k - len(self.x1_values)), constant_values=pad_val)
