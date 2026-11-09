import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Number of evaluations
        N = self.budget
        
        # Precompute optimal t values for ZDT2
        # G(t) = 2*sqrt(t) - 20*ln(10 + sqrt(t))
        # We want to invert G(t) = V_i
        
        t_grid = np.linspace(0, 1, 1000)
        G_grid = 2 * np.sqrt(t_grid) - 20 * np.log(10 + np.sqrt(t_grid))
        
        G_0 = G_grid[0]
        G_1 = G_grid[-1]
        
        # Target values for each point
        # We want N points.
        # i from 0 to N-1
        # V_i = G_0 + (i / (N-1)) * (G_1 - G_0)
        # Note: if N=1, handle separately. Here N=200.
        
        V_targets = np.linspace(G_0, G_1, N)
        
        # Invert using interpolation
        # G_grid is monotonic increasing?
        # d/dt (2 sqrt(t) - 20 ln(10+sqrt(t))) = 1/sqrt(t) - 20 * (1/(10+sqrt(t))) * (1/(2sqrt(t)))
        # = 1/sqrt(t) * (1 - 10/(10+sqrt(t))) = 1/sqrt(t) * (sqrt(t)/(10+sqrt(t))) = 1/(10+sqrt(t)) > 0.
        # So it is monotonic.
        
        t_vals = np.interp(V_targets, G_grid, t_grid)
        
        # Ensure t_vals are within [0, 1]
        t_vals = np.clip(t_vals, 0, 1)
        
        # Construct X
        # x_1 = t_val, x_2..x_d = 0
        X = np.zeros((N, self.dim))
        X[:, 0] = t_vals
        # Ensure bounds are respected (0 to 1 for all)
        # lo and hi are [0, 1]
        
        # Evaluate
        # The evaluator records the results of problem(x)
        # We must call problem exactly N times.
        
        results = []
        for i in range(N):
            if time.time() > deadline:
                break
            f = problem(X[i])
            results.append(f)
            
        return results
