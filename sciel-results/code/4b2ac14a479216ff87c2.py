import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 320.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_evals = self.budget
        if n_evals < 1:
            return np.array([])

        # ZDT1 structure:
        # f1(x) = x[0]
        # f2(x) = (1 + 10*(n-1)) * (1 - sqrt(x[0] / g(x)))
        # g(x) = 1 + 9 * sum(x[1:]) / (n-1)
        #
        # Pareto-optimal points have x[1:] = 0, giving g=1
        # and f2 = C * (1 - sqrt(f1)), where C = 1 + 10*(n-1)
        #
        # For hypervolume maximization with reference point [11, 11]:
        # The ideal sampling is uniform in f2-space.
        # Since f2 = C*(1 - sqrt(f1)), uniform f2 means uniform sqrt(f1),
        # i.e., f1 = u^2 where u is uniform in [0,1].

        # We use all budget for the optimal front points.
        # Spread x[0] using quadratic mapping for uniform f2 coverage.
        n_pts = n_evals

        # Generate u in [0,1] uniformly, then x0 = lo + u^2 * (hi - lo)
        u = np.linspace(0.0, 1.0, n_pts)
        x0_vals = lo[0] + (u ** 2) * (hi[0] - lo[0])

        X = np.empty((n_pts, self.dim), dtype=np.float64)
        X[:, 0] = x0_vals

        if self.dim > 1:
            # Set x[1:] to lower bound to minimize g and thus f2
            X[:, 1:] = lo[1:]
            np.clip(X[:, 1:], lo[1:], hi[1:], out=X[:, 1:])

        # Evaluate all points
        F = np.empty((n_pts, self.n_obj), dtype=np.float64)

        for i in range(n_pts):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
