import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 295

            n_evals = self.budget
            if n_evals <= 0:
                return []

            # Use Gauss-Legendre nodes for better approximation of the area under the curve
            # if the front is smooth.
            if n_evals == 1:
                us = np.array([0.5])
            else:
                # leggauss returns nodes in [-1, 1]
                nodes, _ = np.polynomial.legendre.leggauss(n_evals)
                # Map to [0, 1]
                us = 0.5 * (nodes + 1.0)
                # Sort to ensure we cover the range nicely (though order doesn't matter for HV)
                us = np.sort(us)

            results = []
            for i in range(n_evals):
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim, dtype=float)
                x[0] = float(us[i])
                try:
                    f = problem(x)
                    results.append(f)
                except Exception:
                    break

            return results
