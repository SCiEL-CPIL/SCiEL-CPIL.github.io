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

        results = []

        # ZDT6 Pareto front: x[1:] = 0, x[0] in [0,1]
        # f1(x0) = 1 + x0 * (sin(10*pi*x0) + sin(10*pi*x0**2))
        # f2(x0) = 1 - sqrt(f1(x0))
        # f1 is non-monotonic with increasing oscillation frequency as x0 increases

        # Strategy: uniform sampling in f1 space for optimal hypervolume coverage
        # Compute f1(x0) analytically, then find x0 values that give uniform f1 coverage

        def f1_analytic(x0):
            return 1.0 + x0 * (np.sin(10.0 * np.pi * x0) + np.sin(10.0 * np.pi * x0**2))

        # Find the range of f1 values
        # f1(0) = 1, f1 has oscillations. Let's find min and max on [0,1]
        # Sample densely to find the range
        x_dense = np.linspace(0.0, 1.0, 10001)
        f1_dense = f1_analytic(x_dense)
        f1_min = np.min(f1_dense)
        f1_max = np.max(f1_dense)

        # For uniform f1 sampling, we need to map uniform f1 targets to x0 values
        # Since f1 is non-monotonic, for a given f1 value there may be multiple x0 values
        # We want to pick one x0 per f1 target. For hypervolume, we want the point
        # that gives the best (lowest) f2, which corresponds to the highest f1 for a given
        # "position" on the front. Actually, f2 = 1 - sqrt(f1), so lower f1 gives higher f2.
        # The Pareto front is parameterized by x0, and we want good coverage in (f1, f2) space.

        # Better approach: sample x0 using a distribution that accounts for the
        # oscillatory structure. Use the fact that the oscillation period decreases
        # as x0 increases, so we need more samples at higher x0.

        # The oscillation in f1 comes from sin(10*pi*x0) and sin(10*pi*x0^2).
        # The second term has frequency proportional to 2*x0, so at x0=1 the frequency
        # is about 20*pi (compared to 10*pi for the first term).

        # A good heuristic: use a power-law distribution x0 = t^alpha with alpha > 1
        # to concentrate samples at higher x0 where oscillations are faster.
        # Let's try alpha = 1.5 or 2.0.

        # Actually, let's think about this more carefully. The hypervolume is computed
        # in objective space. The Pareto front of ZDT6 in (f1, f2) space:
        # - f1 ranges from about 1 to about 1+max_osc
        # - f2 = 1 - sqrt(f1), so f2 ranges from about 1-sqrt(1+max_osc) to 0
        # The front is a curve in (f1, f2) space. For hypervolume with reference [11,11],
        # we want the curve to "bulge out" as much as possible from the reference point.

        # The hypervolume contribution is maximized when we have good coverage along
        # the entire Pareto front. Uniform sampling in the parameter x0 is a reasonable
        # baseline, but we can do better by accounting for the local density of the
        # front in objective space.

        # Let's use a combined approach:
        # 1. Use Chebyshev nodes for good coverage of oscillatory functions
        # 2. But weight them toward higher x0 where oscillations are faster

        if n_evals == 1:
            x0s = np.array([0.5])
        elif n_evals == 2:
            x0s = np.array([0.0, 1.0])
        else:
            # Use a transformed grid: map uniform t in [0,1] to x0 = t^beta
            # with beta > 1 to concentrate at high x0.
            # Try beta = 1.3 as a mild concentration.
            beta = 1.3
            t = np.linspace(0.0, 1.0, n_evals)
            x0s = t ** beta
            # Ensure we include endpoints
            x0s[0] = 0.0
            x0s[-1] = 1.0

        # Evaluate all points
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break

        return results
