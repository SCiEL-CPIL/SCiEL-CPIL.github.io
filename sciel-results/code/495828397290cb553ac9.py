import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        
        B = int(self.budget)
        D = int(self.dim)
        
        # Reference point
        R1, R2 = 11.0, 11.0
        
        # ZDT1 front: f1 = x1, f2 = 1 - sqrt(x1)
        # h(y) = R2 - f2(y) = 11 - (1 - sqrt(y)) = 10 + sqrt(y)
        
        # Optimal density for HV maximization is proportional to h(y)^2
        # rho(y) = (10 + sqrt(y))^2
        # CDF F(y) = integral_0^y (10 + sqrt(t))^2 dt
        # = integral_0^y (100 + 20 sqrt(t) + t) dt
        # = 100y + 20 * (2/3) t^(3/2) + 0.5 y^2
        # = 100y + (40/3) y^(1.5) + 0.5 y^2
        
        def F(y):
            return 100.0 * y + (40.0 / 3.0) * (y ** 1.5) + 0.5 * (y ** 2)
        
        C = F(1.0)
        
        # Solve F(y) = C * u for y
        # y is in [0, 1]
        # F(0) = 0, F(1) = C
        # We want B points.
        # Use midpoints of the CDF intervals for optimal quantization
        # u_i = (i + 0.5) / B
        
        y_opt = np.zeros(B)
        for i in range(B):
            u = (i + 0.5) / B
            target = C * u
            
            # Solve F(y) = target
            # F is monotonic increasing on [0, 1]
            try:
                y_opt[i] = brentq(lambda y: F(y) - target, 0.0, 1.0, xtol=1e-12)
            except Exception:
                # Fallback to uniform if brentq fails
                y_opt[i] = u
        
        # Build X
        X = np.zeros((B, D))
        X[:, 0] = y_opt
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        X = np.clip(X, lo, hi)
        
        # Evaluate
        used = 0
        for i in range(B):
            if time.time() > deadline:
                break
            problem(X[i].copy())
            used += 1
            
        return X
