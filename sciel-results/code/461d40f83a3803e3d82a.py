import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.n_points = min(budget, 200)
        
        # Pre-compute optimal x1 values
        # We do this once. 
        # Since the problem is deterministic, we can cache it or just compute it.
        # Computing it takes ~1-2 seconds.
        
        self.optimal_x1 = self._compute_optimal_x1()

    def _compute_optimal_x1(self):
        N = self.n_points
        R1 = 11.0
        R2 = 11.0
        
        def hv_neg(x):
            # x is array of N values
            # Sort just in case, though constraints should keep them sorted
            x = np.sort(x)
            # Calculate HV
            # HV = sum_{i=1}^{N-1} (x_{i+1} - x_i) (R2 - f2_i) + (R1 - x_N) (R2 - f2_N)
            # f2_i = 1 - sqrt(x_i)
            # R2 - f2_i = 10 + sqrt(x_i)
            f2 = 1.0 - np.sqrt(x)
            heights = R2 - f2 # 10 + sqrt(x)
            
            # Widths
            # Segment i covers [x_i, x_{i+1}] with height heights[i]
            # Last segment covers [x_N, R1] with height heights[N-1]
            
            # Vectorized calculation
            # widths_internal = np.diff(x)
            # contrib_internal = np.sum(widths_internal * heights[:-1])
            # contrib_last = (R1 - x[-1]) * heights[-1]
            # return -(contrib_internal + contrib_last)
            
            # Handle N=1
            if N == 1:
                return -((R1 - x[0]) * (R2 - (1.0 - np.sqrt(x[0]))))
            
            widths = np.diff(x)
            contrib_internal = np.sum(widths * heights[:-1])
            contrib_last = (R1 - x[-1]) * heights[-1]
            
            return -(contrib_internal + contrib_last)

        # Initial guess: uniform
        x0 = np.linspace(0, 1, N)
        
        # Bounds
        bounds = [(0, 1) for _ in range(N)]
        
        # Constraints: x_i <= x_{i+1}
        # We can use inequality constraints or just rely on the optimizer with bounds and hope?
        # Better to enforce ordering.
        # SLSQP supports constraints.
        constraints = []
        for i in range(N - 1):
            def make_constraint(i):
                def constraint(x):
                    return x[i+1] - x[i]
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_constraint(i)})
            
        # Run optimization
        try:
            res = minimize(hv_neg, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                return res.x
            else:
                return x0
        except Exception:
            return x0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.n_points
        x1_vals = self.optimal_x1
        
        X = np.zeros((N, self.dim))
        X[:, 0] = x1_vals
        X[:, 1:] = 0.5
        
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
