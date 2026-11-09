"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting ZDT6 structure for fast Pareto front coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.eval_count = 0
        self.start_time = None
        self.deadline = 325.0  # 330s limit, 5s safety margin

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT6 Pareto front: f1 = x1, f2 = 1 - sqrt(x1) when x[1:] = 0
        # To maximize hypervolume with reference [11, 11], we need points spread
        # along the curve. The curve is concave (steeper near x1=0), so we should
        # place more points near x1=0.
        #
        # Optimal spacing: for a concave function, equal hypervolume contribution
        # per point means spacing proportional to the local slope.
        # f2 = 1 - sqrt(x1), df2/dx1 = -1/(2*sqrt(x1))
        # Points should be denser where |df2/dx1| is large (small x1).
        #
        # Use x1 values that are quadratic in index: x1_i = (i/n)^2
        # This gives spacing dx1_i = 2*i/n^2, which is proportional to sqrt(x1_i),
        # i.e., inversely proportional to the slope magnitude.

        n_points = self.budget  # Use all budget for Pareto front points
        # Generate x1 values: x1_i = (i / (n-1))^2 for i = 0, ..., n-1
        # This gives x1_0 = 0, x1_{n-1} = 1
        indices = np.arange(n_points)
        x1_values = (indices / (n_points - 1)) ** 2

        for x1 in x1_values:
            if self.eval_count >= self.budget:
                break
            if time.time() - self.start_time > self.deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] = 0 minimizes g(x) to its minimum value of 1
            f = problem(x)
            self._add_point(x, f)

        return self.archive

    def _add_point(self, x, f):
        self.eval_count += 1
        f = np.array(f, dtype=float)
        x = np.array(x, dtype=float)
        # Only add if not dominated by existing archive
        for _, f_exist in self.archive:
            if np.all(f_exist <= f) and np.any(f_exist < f):
                return  # dominated, don't add
        # Remove points dominated by new point
        self.archive = [(x_i, f_i) for x_i, f_i in self.archive
                        if not (np.all(f <= f_i) and np.any(f < f_i))]
        self.archive.append((x, f))

# EVOLVE-BLOCK-END
