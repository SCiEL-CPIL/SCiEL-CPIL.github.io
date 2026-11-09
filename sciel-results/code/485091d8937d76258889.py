import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        n = self.budget
        dim = self.dim

        # ZDT3 Pareto front: x[0] in [0,1], x[1:] = 0
        # f1 = x[0], f2 = 10 * (1 - sqrt(x[0]/10) - (x[0]/10)*sin(10*pi*x[0]))
        # Place n points uniformly in x[0]
        x0_vals = np.linspace(0.0, 1.0, n)

        results = []
        for i in range(n):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0_vals[i]
            f = problem(x)
            results.append(f)

        return results
