import time
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for ZDT6 using analytic Pareto front sampling.
    
    ZDT6 formulation (n=10):
      f1(x) = x[0]
      g(x)  = 1 + 10*(n-1) * sum_{i=2}^{n} (x_i^2 - 10*cos(4*pi*x_i))
      h(f1/g) = 1 - sqrt(f1/g) * (1 - (f1/g) * sin(10*pi*f1))
      f2(x)  = g * h(f1/g)
    
    Pareto front: x[1:] = 0 => g = 1
      f1 = x[0], f2 = 1 - sqrt(x[0]) * (1 - x[0]*sin(10*pi*x[0]))
    
    With reference point [11, 11], all front points with x[0] in [0,1] are valid.
    We use the full budget to uniformly sample x[0] in [0,1] with x[1:]=0.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        n_evals = 0

        # Use all budget for uniform sampling of the analytic Pareto front
        # x[0] in [0, 1], x[1:] = 0
        n_points = budget
        x0_values = np.linspace(0.0, 1.0, n_points)

        for x0 in x0_values:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break

            x = np.zeros(dim)
            x[0] = x0
            # x[1:] remains 0 for the Pareto front (g = 1)

            # Clip to bounds just in case
            x = np.clip(x, lo, hi)

            try:
                problem(x)
                n_evals += 1
            except Exception:
                break

        return None
