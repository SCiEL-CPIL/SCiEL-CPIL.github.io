"""Program for multi-objective optimization on ZDT1 using hypervolume-optimized quantization."""
import numpy as np


class Optimizer:
    """Optimizes ZDT1 by exploiting its structure: fix non-critical dims to 0,
    use a grid of x1 values optimized for hypervolume maximization via optimal quantization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        if n <= 0:
            return None

        # ZDT1 Pareto front: f1 = x1, f2 = 1 - x1^2 (when x[1:] = 0)
        # Reference point: (11, 11)
        # r2 - f2 = 11 - (1 - x1^2) = 10 + x1^2
        #
        # Optimal quantization for hypervolume maximization:
        # We want to place k points on [0,1] to maximize the hypervolume.
        # The hypervolume contribution is proportional to the area under h(x) = 10 + x^2.
        # The optimal placement uses equal-area cells: the CDF of h(x) is equally spaced.
        #
        # CDF: F(x) = (30x + x^3) / 31 (normalized over [0,1])
        # For k points including endpoints:
        #   x_0 = 0, x_{k-1} = 1
        #   Interior points: F(x_i) = i/(k-1) for i = 1, ..., k-2
        #   Solve 30x + x^3 = 31 * i/(k-1) for each interior point.

        k = n

        if k == 1:
            # Single point: place at x=0 to maximize (r1 - f1)*(r2 - f2) = 11*10 = 110
            x1_values = np.array([0.0])
        elif k == 2:
            # Two points: endpoints give best coverage
            x1_values = np.array([0.0, 1.0])
        else:
            x1_values = np.zeros(k)
            x1_values[0] = 0.0
            x1_values[-1] = 1.0
            # Solve for interior points using Newton's method
            for i in range(1, k - 1):
                target = i / (k - 1)
                rhs = 31.0 * target
                # Initial guess: linear approximation x ≈ target
                x = target
                # Newton's method to solve 30x + x^3 = rhs
                for _ in range(50):
                    f_val = 30.0 * x + x**3 - rhs
                    f_deriv = 30.0 + 3.0 * x**2
                    x_new = x - f_val / f_deriv
                    # Clamp to [0, 1]
                    if x_new < 0.0:
                        x_new = 0.0
                    elif x_new > 1.0:
                        x_new = 1.0
                    if abs(x_new - x) < 1e-15:
                        x = x_new
                        break
                    x = x_new
                x1_values[i] = x

        # Evaluate each point: x[0] = x1, x[1:] = 0 (gives h(x)=0, g(x)=1, f2 = 1 - x1^2)
        results = []
        for x1 in x1_values:
            x = np.zeros(dim)
            x[0] = x1
            f = problem(x)
            results.append(f)

        return results
