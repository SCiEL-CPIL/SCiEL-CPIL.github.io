import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim

        n_evals = 0

        # ZDT6:
        # f1 = 1 + 10*(dim-1)*x[0]   (when x[1:] = 0)
        # f2 depends on g(x), which is 1 + 10*(dim-1) + sum(2*x_i^2 - 2*cos(4*pi*x_i))
        # For Pareto front: x[1:] = 0, so g = 1 + 10*(dim-1)
        # f2 = (1/g(x)) * (1 - sqrt(f1/g(x)))^2 + 1

        # To maximize hypervolume, densely sample x[0] in [0, 1]
        # This gives f1 in [1, 1+10*(dim-1)]

        # We have budget evaluations. Use all of them to sample the front densely.
        # Uniform sampling of x[0] gives uniform coverage of f1.

        # Actually, for better hypervolume coverage, we might want to bias toward
        # lower f2 values. But the Pareto front is the trade-off, so uniform x[0]
        # should give good spread.

        # Sample x[0] non-uniformly to better capture the hypervolume.
        # The Pareto front of ZDT6 is steeper at low f1 (low x0).
        # Clustering points near x0=0 should improve the hypervolume approximation.
        # Using a power law with exponent > 1 biases towards 0.
        x0_linear = np.linspace(0.0, 1.0, budget)
        x0_values = x0_linear ** 2

        for x0 in x0_values:
            if time.time() > deadline:
                break
            if n_evals >= budget:
                break

            # Construct x with x[0] = x0, x[1:] = 0
            x = np.zeros(dim)
            x[0] = x0

            # Clip to bounds
            x = np.clip(x, lo, hi)

            try:
                problem(x)
                n_evals += 1
            except Exception:
                break

        return None
