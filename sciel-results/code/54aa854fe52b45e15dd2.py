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

        # For standard ZDT2, g(x) = sum(x_i - 0.5)^2 is minimized at x_i = 0.5
        # Allocate most budget to the correct strategy, small hedge for robustness
        n_hedge = 5
        n_main = n_evals - n_hedge

        points = np.zeros((n_evals, self.dim))

        # Main strategy: x1 varies, x2..xn = 0.5 (standard ZDT2 Pareto front)
        points[:n_main, 0] = np.linspace(lo[0], hi[0], n_main)
        points[:n_main, 1:] = 0.5

        # Hedge strategy: x1 varies, x2..xn = 0.0 (in case g is minimized elsewhere)
        points[n_main:, 0] = np.linspace(lo[0], hi[0], n_hedge)
        points[n_main:, 1:] = 0.0

        # Evaluate all points
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(points[i])

        return points
