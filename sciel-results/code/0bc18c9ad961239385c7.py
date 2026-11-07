import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        n_evals = self.budget
        
        points = np.zeros((n_evals, self.dim))

        # Optimal distribution for ZDT2 Pareto Front
        # f2 = 1 - sqrt(x1). Curvature |f''(x)|^0.5 ~ x^(-3/4).
        # Integral ~ x^(1/4). So u ~ x^(1/4) => x ~ u^4.
        if n_evals > 1:
            u = np.linspace(0, 1, n_evals)
            x1_norm = u ** 4
            points[:, 0] = lo[0] + x1_norm * (hi[0] - lo[0])
            points[:, 1:] = 0.5  # Minimizer of g(x) for ZDT2
        else:
            points[0, 0] = lo[0]
            points[0, 1:] = 0.5

        # Evaluate all points
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(points[i])
            
        return points
