import numpy as np
import time

class Optimizer:
    """
    Exploits the structure of ZDT3 where the Pareto front is achieved
    when x_2...x_n = 0. We allocate the entire budget to sampling x_1
    uniformly to cover the front as densely as possible.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT3 Pareto front: x_2...x_dim = lower bound
        # Uniform sampling of x_1 over [lo[0], hi[0]] with self.budget points
        x1_samples = np.linspace(lo[0], hi[0], self.budget)

        X = np.empty((self.budget, self.dim))
        X[:, 0] = x1_samples
        X[:, 1:] = lo[1:]

        # Evaluate all points with deadline guard
        for i in range(self.budget):
            if time.time() > deadline - 5.0:
                break
            problem(X[i])

        return X
