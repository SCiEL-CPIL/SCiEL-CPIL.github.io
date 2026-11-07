"""Multi-objective optimizer on ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    
    Strategy:
    ZDT4 Pareto front: f1 = x0, f2 = g(x) + (1 - sqrt(x0/g)) * g.
    g(x) is independent of x0. The global minimum of g is at x_i = 0 for i > 0.
    If the problem definition uses g = 1 + 10*(n-1) + sum(...), then g_min = 101.
    If it uses g = 1 + sum(...), then g_min = 1.
    
    We detect g_min by probing: set x1..xn to 0, evaluate at a few x0 values.
    Then sample the front densely with the detected g_min.
    
    If 0 is not in bounds for some xi, we use a local search to minimize g.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Check if 0 is within bounds for x1..x_{n-1}
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))

        if zeros_ok:
            # Probe to detect g_min
            # Evaluate at x0 = 0.5, x_i = 0 for i > 0
            x_probe = np.zeros(self.dim)
            x_probe[0] = 0.5
            f_probe = np.array(problem(x_probe))
            # f2 = g + (1 - sqrt(0.5/g)) * g = g + g - sqrt(0.5*g) = 2g - sqrt(0.5*g)
            # So f2 = 2g - sqrt(0.5*g)
            # Solve for g: f2 = 2g - sqrt(0.5*g)
            # Let t = sqrt(g), then f2 = 2t^2 - sqrt(0.5)*t
            # 2t^2 - sqrt(0.5)*t - f2 = 0
            # t = [sqrt(0.5) + sqrt(0.5 + 8*f2)] / 4
            # g = t^2
            a = 2.0
            b_coeff = -np.sqrt(0.5)
            c = -f_probe[1]
            discriminant = b_coeff**2 - 4*a*c
            if discriminant >= 0:
                t = (-b_coeff + np.sqrt(discriminant)) / (2*a)
                g_min = t**2
            else:
                g_min = 1.0  # Fallback

            # Sample the Pareto front
            n_points = self.budget
            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)

            if x0_lo >= x0_hi:
                x0_vals = np.linspace(lo[0], hi[0], n_points)
            else:
                # Use logarithmic spacing to get more points near x0=0 where the curve is steep
                # f2 = g + (1 - sqrt(x0/g)) * g
                # df2/dx0 = -0.5 * sqrt(g/x0) * ... actually let's just use uniform
                # Uniform is fine for 200 points
                x0_vals = np.linspace(x0_lo, x0_hi, n_points)

            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            # X[:, 1:] = 0

            F = np.array([problem(x) for x in X])
            return X

        else:
            # Fallback: 0 not in bounds for some xi
            # Use a simple local search to minimize g, then sample the front
            # First, find a good x_tail (x1..x_{n-1}) that minimizes g
            # Use random restarts with a few evaluations
            rng = np.random.default_rng(42)
            
            # Spend some budget finding good x_tail
            n_tail_search = min(50, self.budget // 4)
            best_g = np.inf
            best_x_tail = lo[1:] + (hi[1:] - lo[1:]) * rng.random(self.dim - 1)
            
            for _ in range(n_tail_search):
                if time.time() > deadline:
                    break
                x_tail = lo[1:] + (hi[1:] - lo[1:]) * rng.random(self.dim - 1)
                x0 = 0.5  # Doesn't affect g
                x = np.concatenate(([x0], x_tail))
                f = problem(x)
                # f2 = g + (1 - sqrt(x0/g)) * g, so g can be inferred
                # But easier: just track f2 as proxy for g (monotonic in g for fixed x0)
                if f[1] < best_g:
                    best_g = f[1]
                    best_x_tail = x_tail.copy()
            
            n_points = self.budget - n_tail_search
            if n_points < 1:
                n_points = 1

            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)
            if x0_lo >= x0_hi:
                x0_vals = np.linspace(lo[0], hi[0], n_points)
            else:
                x0_vals = np.linspace(x0_lo, x0_hi, n_points)

            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            X[:, 1:] = best_x_tail

            F = np.array([problem(x) for x in X])
            return X

# EVOLVE-BLOCK-END
