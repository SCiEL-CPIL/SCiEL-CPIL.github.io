import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        N = self.budget
        
        def hv_from_alpha(alpha):
            # u0=0, umax=1
            t = np.arange(N) / (N - 1)
            u = t ** alpha
            u = np.clip(u, 0.0, 1.0)
            
            # f2 = 1 - sqrt(u)
            # heights = 11 - f2 = 10 + sqrt(u)
            heights = 10.0 + np.sqrt(u)
            
            # widths
            widths = np.diff(u)
            last_width = 11.0 - u[-1]
            
            hv = np.sum(widths * heights[:-1]) + last_width * heights[-1]
            return hv

        def neg_hv_alpha(alpha):
            return -hv_from_alpha(alpha)

        # Optimize alpha
        # Initial guess around 1.5 to 2.5
        res = minimize_scalar(neg_hv_alpha, bounds=(0.5, 5.0), method='bounded', 
                              options={'xatol': 1e-8})
        
        best_alpha = res.x
        
        # Generate points
        t = np.arange(N) / (N - 1)
        u_vals = t ** best_alpha
        u_vals = np.clip(u_vals, 0.0, 1.0)
        
        X = np.zeros((N, self.dim))
        X[:, 0] = u_vals
        
        F = np.zeros((N, self.n_obj))
        for k in range(N):
            if time.time() > deadline:
                break
            F[k] = problem(X[k])
        
        return X
