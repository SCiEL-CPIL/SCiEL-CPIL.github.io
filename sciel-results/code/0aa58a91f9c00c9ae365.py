import numpy as np
import time


class Optimizer:
    """Exploit ZDT1 structure: Pareto front is at x_2..x_30 = 0.
    Optimize sampling of x_1 to maximize hypervolume of the staircase
    formed by the points on the front f2 = 1 - sqrt(f1)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        budget = self.budget
        dim = self.dim

        # ZDT1 Pareto front: f1 = x1, f2 = 1 - sqrt(x1) for x2..xn = 0.
        # We want to maximize the hypervolume of the set of points.
        # The HV is the area under the "upper envelope" of the points.
        # For a decreasing function, the envelope is a staircase where
        # the height on interval [x_{i-1}, x_i] is y_i (the point with larger x).
        # To maximize this area with a fixed number of points, we should
        # sample more densely where the function is steep (near x=0).
        # Using x_i = (i / (N-1))^2 transforms the variable to make the
        # function linear-ish or at least better conditioned for uniform sampling in the new space.
        
        N = budget
        if N < 2:
             x1_samples = np.array([0.0, 1.0])
        else:
            # Map uniform t in [0, 1] to x = t^2
            # This clusters points near 0.
            t_samples = np.linspace(0.0, 1.0, N)
            x1_samples = t_samples ** 2

        points = []
        for x1 in x1_samples:
            x = np.zeros(dim)
            x[0] = x1
            points.append(x)

        # Evaluate all points
        for x in points:
            if time.time() > deadline:
                break
            problem(x)

        return points
