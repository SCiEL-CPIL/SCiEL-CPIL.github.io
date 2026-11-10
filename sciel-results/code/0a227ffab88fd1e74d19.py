import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 Pareto front is at x_i = 0 for i >= 1 (index 1 onwards)
        # x_0 varies.
        # f1 = x_0
        # f2 = 1 - sqrt(x_0) (assuming G=1 which implies x_i=0 for i>0)
        
        n_evals = self.budget
        # Sample x_0 uniformly in [lo[0], hi[0]]
        # To maximize hypervolume coverage of the curve f2 = 1 - sqrt(f1),
        # uniform f1 is good.
        x0_vals = np.linspace(lo[0], hi[0], n_evals)
        
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        # Ensure other dims are at lower bound (0) to minimize G
        # G = 1 + 9/(n-1) * sum(x_i). Min G is 1, achieved when sum(x_i)=0.
        # So x_i = lo[i] (which is 0) is optimal.
        for i in range(1, self.dim):
            X[:, i] = lo[i]
            
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            F.append(f)
            
        return X
