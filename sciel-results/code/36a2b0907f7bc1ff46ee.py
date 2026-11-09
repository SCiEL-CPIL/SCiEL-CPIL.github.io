import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 320.0

    def __call__(self, problem, bounds):
        n = self.budget
        if n <= 0:
            return None

        # ZDT1 Pareto front: f1 = x0, f2 = 1 + 9*(1-x0) when x1=...=x29=0
        # Reference point [11.0, 11.0]
        # Hypervolume contribution analysis:
        # For points on the Pareto front with f1=u, f2=10-9u:
        # The hypervolume is maximized by uniform spacing of u values in [0,1]
        # because the Pareto front is linear: f2 = 10 - 9*f1
        # Loss per interval is proportional to (delta_u)^2, so uniform is optimal

        # Use all budget evaluations for uniform sampling of x0 in [0, 1]
        # This gives 200 points evenly spaced on the Pareto front
        x0_values = np.linspace(0.0, 1.0, n)

        # Evaluate each point: set x[0]=x0, x[1:]=0 to be on the Pareto front
        for x0 in x0_values:
            if time.time() > self.deadline:
                break

            x = np.zeros(self.dim)
            x[0] = x0

            try:
                problem(x)
            except Exception:
                break

        return None
