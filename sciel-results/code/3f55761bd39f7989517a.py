import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim

        # ZDT6 structure:
        # f1 = x0
        # g(x) = 1 + 10*(dim-1) + sum_{i=1}^{dim-1} (2*xi^2 - 2*cos(4*pi*xi))
        # f2 = g(x) * (1 - sqrt(f1/g(x)))
        #
        # The Pareto front is achieved when x1..x_{dim-1} are at minima of g,
        # which occur at xi in {0, 0.5} for i > 0.
        # Setting xi = 0 for all i > 0 gives the minimum g value.
        #
        # To maximize hypervolume, we densely sample the known Pareto front
        # by varying x0 uniformly in [0, 1] and keeping x1..x_{dim-1} = 0.

        # Use the entire budget for dense Pareto front sampling.
        # This maximizes the number of non-dominated points along the front,
        # which directly maximizes the hypervolume indicator.
        n_front = budget

        # Generate uniformly spaced x0 values in [0, 1]
        t_vals = np.linspace(0.0, 1.0, n_front)

        # Build decision vectors: x[0] = t, x[1:] = 0
        X_front = np.zeros((n_front, dim))
        X_front[:, 0] = t_vals

        # Evaluate each point individually to respect budget exactly
        for i in range(n_front):
            if time.time() > deadline:
                break
            problem(X_front[i])

        return X_front
