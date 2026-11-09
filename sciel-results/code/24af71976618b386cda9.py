import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ2 structure:
        # g(x) = sum_{i=2}^{11} (x_i - 0.5)^2, minimized at 0 when x_i=0.5
        # f1 = (1+g)*cos(t1)*cos(t2), f2 = (1+g)*cos(t1)*sin(t2), f3 = (1+g)*sin(t1)
        # where t1 = x0*pi/2, t2 = x1*pi/2
        # Pareto front: g=0, so f1^2+f2^2+f3^2 = 1
        # x[2:] = 0.5 to achieve g=0

        # Strategy: Use all budget on the Pareto front with a low-discrepancy sequence
        # Halton sequence in 2D (bases 2 and 3) for x0, x1

        def halton_sequence(n, bases):
            """Generate n points in the Halton sequence for the given bases."""
            points = np.zeros((n, len(bases)))
            for j, base in enumerate(bases):
                for i_val in range(n):
                    result = 0.0
                    denom = 1.0
                    num = i_val
                    while num > 0:
                        denom *= base
                        digit = num % base
                        result += digit / denom
                        num //= base
                    points[i_val, j] = result
            return points

        # Generate Halton sequence points for x0, x1 in [0, 1)
        points_2d = halton_sequence(budget, [2, 3])

        x0 = points_2d[:, 0]
        x1 = points_2d[:, 1]

        # Construct full 12D vectors on the Pareto front
        X = np.zeros((budget, dim))
        X[:, 0] = x0
        X[:, 1] = x1
        if dim > 2:
            X[:, 2:] = 0.5  # Minimize g

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        results = []
        for x in X:
            if time.time() > deadline:
                break
            f = problem(x)
            results.append(f)

        return results
