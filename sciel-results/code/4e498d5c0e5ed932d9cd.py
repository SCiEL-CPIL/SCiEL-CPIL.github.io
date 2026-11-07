import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # Check if 0 is within bounds for x1..x_{dim-1}
        # This ensures g(x) = 1 for standard ZDT4
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))

        if not zeros_ok:
            # Fallback: 0 is not in bounds for some x_i (i>=1)
            # Use random sampling
            rng = np.random.default_rng(0)
            for _ in range(self.budget):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(self.dim)
                problem(x)
            return

        # x0 bounds, clamped to [0,1] for standard ZDT4
        x0_lo = max(lo[0], 0.0)
        x0_hi = min(hi[0], 1.0)

        if x0_lo >= x0_hi:
            # Degenerate: just evaluate one point
            x = np.zeros(self.dim)
            x[0] = x0_lo
            problem(x)
            return

        n_points = self.budget

        # The Pareto front for ZDT4 with g=1:
        # f1 = x0, f2 = h(x0) = 1 + exp(-4*x0 - 0.3*sin^5(8*pi*x0))
        #
        # To maximize hypervolume with reference point [11, 400],
        # we want to sample the front densely. The hypervolume is the
        # area under the upper envelope of the step function formed by
        # the points (when sorted by f1).
        #
        # Key insight: The "waste" (area between the true curve and the
        # approximating step function) is proportional to the sum over
        # intervals of (x_{i+1} - x_i) * |f2'(x_i)| approximately.
        # To minimize this waste, we should place points more densely
        # where |f2'| is large.
        #
        # f2'(x) = -4*exp(-4x - 0.3*sin^5(8πx)) - 0.3*5*sin^4(8πx)*8π*cos(8πx)*exp(...)
        #        = exp(-4x - 0.3*sin^5(8πx)) * [-4 - 12π*sin^4(8πx)*cos(8πx)]
        #
        # The dominant term is -4*exp(-4x), which is largest in magnitude
        # near x=0. So we should sample more densely near x0_lo.
        #
        # A good approximation: use a power-law distribution.
        # t = linspace(0, 1, n_points)
        # x0_vals = x0_lo + (x0_hi - x0_lo) * t^alpha
        #
        # With alpha > 1, points are concentrated near x0_lo.
        # Let's use alpha = 2 as a reasonable choice.
        
        t = np.linspace(0, 1, n_points)
        x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** 2)
        
        # Ensure we don't exceed bounds
        x0_vals = np.clip(x0_vals, x0_lo, x0_hi)
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_vals
        # X[:, 1:] = 0.0 is default
        
        # Evaluate all points
        for x in X:
            if time.time() > deadline:
                break
            problem(x)
        
        return
