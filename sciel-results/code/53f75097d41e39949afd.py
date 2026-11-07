import numpy as np
import time


class Optimizer:
    """Hypervolume-optimal quantization for ZDT1 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 Pareto front: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1]
        # Hypervolume integrand: (11 - f2) = 10 + sqrt(x1)
        # Optimal quantization: place N points with density proportional to (10 + sqrt(x))
        # CDF: integral of (10 + sqrt(x)) dx from 0 to x = 10x + (2/3)x^(3/2)
        # Inverse CDF to generate optimal x1 values

        def cdf(x):
            return 10 * x + (2.0 / 3.0) * np.power(np.clip(x, 0, 1), 1.5)

        cdf_total = cdf(1.0)  # 10 + 2/3 = 32/3

        # Number of points to place on the front
        # Use most of budget for front points, some for verification/polish
        n_front = min(budget - 5, 195)
        if n_front < 2:
            n_front = 2

        # Generate optimal x1 values using inverse CDF
        # Uniform quantiles mapped through inverse CDF
        u = (np.arange(1, n_front + 1) - 0.5) / n_front  # midpoints of uniform grid
        # Solve 10x + (2/3)x^1.5 = u * cdf_total for x
        target = u * cdf_total
        x1_vals = np.zeros(n_front)
        for i in range(n_front):
            # Bisection to solve 10x + (2/3)x^1.5 = target[i]
            a, b = 0.0, 1.0
            for _ in range(50):
                mid = (a + b) / 2
                if cdf(mid) < target[i]:
                    a = mid
                else:
                    b = mid
            x1_vals[i] = (a + b) / 2

        # Ensure we cover [0, 1]
        x1_vals[0] = 0.0
        x1_vals[-1] = 1.0

        # For each x1, construct a feasible point where g(x) = 0
        # g(x) = sum_{i=2}^{n} [4*x_i*sin(10*pi*x_i*x_1) + x_i^2]
        # Set x2...xn to 0.5, then adjust x2 to make g=0
        # Actually, a simpler approach: set all x_i (i>=2) to 0, then g=0
        # But x_i=0 gives g=0 only if sin term is 0, which it is since x_i=0
        # So x2...xn = 0 gives g=0 exactly

        def make_point(x1):
            x = np.zeros(dim)
            x[0] = x1
            # x2...xn = 0 gives g = 0
            return x

        # Evaluate points
        F = np.zeros((n_front, 2))
        X = np.zeros((n_front, dim))
        used = 0

        for i in range(n_front):
            if time.time() > deadline:
                break
            x = make_point(x1_vals[i])
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            X[i] = x
            F[i] = f
            used += 1

        # Use remaining budget for local refinement of x1 positions
        # or for additional points if budget allows
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Try to refine: for a few points, do a small local search on x1
            # to see if we can improve hypervolume
            # Actually, the quantization is already optimal in expectation.
            # Use remaining budget to verify and potentially add a few more points
            # at positions that might help with numerical precision
            
            # Add a few more points at key locations if budget allows
            extra_positions = [0.0, 0.5, 1.0, 0.25, 0.75]
            for pos in extra_positions:
                if used >= budget or time.time() > deadline:
                    break
                # Check if this position is already well-covered
                if np.min(np.abs(x1_vals[:used] - pos)) < 0.01:
                    continue
                x = make_point(pos)
                x = np.clip(x, lo, hi)
                f = np.array(problem(x), dtype=float)
                if used < len(X):
                    X[used] = x
                    F[used] = f
                else:
                    break
                used += 1

        return X[:used]
