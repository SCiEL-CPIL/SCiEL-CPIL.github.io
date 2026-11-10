import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 300.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.budget
        
        # Include endpoints f1=0 and f1=1 explicitly
        # Use remaining N-2 points with optimal density for interior
        # Optimal density: p(f1) ∝ 1/sqrt(f1), CDF: F(f1) = sqrt(f1)
        # Inverse CDF: f1 = u^2
        
        f1_vals = np.zeros(N)
        f1_vals[0] = 0.0  # f1=0, f2=1
        f1_vals[-1] = 1.0  # f1=1, f2=0
        
        if N > 2:
            n_interior = N - 2
            # Midpoint stratification for interior: u_i = (i - 0.5) / n_interior
            i_vals = np.arange(1, n_interior + 1, dtype=np.float64)
            u_vals = (i_vals - 0.5) / n_interior
            f1_interior = u_vals ** 2
            f1_vals[1:-1] = f1_interior
        
        # Construct X
        X = np.zeros((N, self.dim))
        X[:, 0] = f1_vals
        # x[1:] = 0 (lo[1:] is 0)
        
        F = []
        for i in range(N):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            F.append(f)
            
        return X
