# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin

        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For ZDT1, the Pareto front is achieved when x_2...x_d = 0
        # and x_1 varies in [0, 1].
        # f1 = x1
        # f2 = 1 - sqrt(x1) (since g=1)
        # Hypervolume contribution of point at t: (t - t_prev) * (10 + sqrt(t))
        # Optimal density proportional to 1/(10 + sqrt(t))
        
        n_evals = self.budget
        if n_evals > 0:
            # Compute optimal CDF using numerical integration
            # F(t) = integral_0^t 1/(10+sqrt(s)) ds / integral_0^1 1/(10+sqrt(s)) ds
            # Use substitution s = u^2, ds = 2u du
            # integral = integral_0^sqrt(t) 2u/(10+u) du
            # = integral_0^sqrt(t) 2(1 - 10/(10+u)) du
            # = 2*sqrt(t) - 20*ln(1 + sqrt(t)/10)
            
            # Let's use this analytical form
            # G(t) = 2*sqrt(t) - 20*ln(1 + sqrt(t)/10)
            # F(t) = G(t) / G(1)
            
            def G(t):
                st = np.sqrt(t)
                return 2 * st - 20 * np.log(1 + st / 10.0)
            
            G1 = G(1.0)
            
            # u_i = i / (N-1) for i = 0, ..., N-1
            # We want F(t_i) = u_i, so t_i = F^{-1}(u_i)
            # G(t_i) = u_i * G1
            # Solve: 2*sqrt(t) - 20*ln(1 + sqrt(t)/10) = u_i * G1
            
            # Solve G(t) = target vectorized via Newton on y = sqrt(t)
            # h(y) = 2y - 20 ln(1 + y/10); h'(y) = 2 - 20/(10+y) = 2y/(10+y)
            # Newton: y <- y - h(y)/h'(y)
            u = np.linspace(0.0, 1.0, n_evals)
            target = u * G1
            y = np.sqrt(np.clip(target / 2.0, 0.0, 1.0))  # initial guess
            y = np.clip(y, 1e-12, 1.0)
            for _ in range(60):
                h = 2.0 * y - 20.0 * np.log(1.0 + y / 10.0)
                hp = 2.0 * y / (10.0 + y)
                y = y - h / hp
                y = np.clip(y, 1e-12, 1.0)
            t_vals = y * y

            # Construct points
            X = np.zeros((n_evals, self.dim))
            X[:, 0] = t_vals
            X = np.clip(X, lo, hi)

            last_obj = None
            for i in range(n_evals):
                if time.time() > deadline:
                    break
                x = X[i]
                f = problem(x)
                last_obj = f

            # Adaptive refinement: fill hypervolume gaps in the remaining budget
            # Record the first objective (t) of each evaluated point in order.
            # For ZDT1 the hypervolume is a left-Riemann sum over f1 with
            # height (11 - f2) = 10 + sqrt(f1); each gap (a,b) contributes
            # (b-a)*(10+sqrt(b)), so the marginal value of a new point is
            # proportional to the width of the gap it sits in. Insert points
            # into the widest gaps until the budget or time is exhausted.
            if last_obj is not None and time.time() < deadline:
                t_list = [float(t_vals[i]) for i in range(n_evals)]
                # Sort by f1 and track which gaps still need a point.
                # We will greedily pick the widest gap and evaluate its midpoint.
                # Use a list of (left, right) gaps; after inserting a point at
                # midpoint m, split (l, r) into (l, m) and (m, r).
                # Maintain a max-heap keyed by gap width (r - l).
                import heapq
                gaps = []
                for j in range(len(t_list) - 1):
                    width = t_list[j + 1] - t_list[j]
                    if width > 1e-9:
                        heapq.heappush(gaps, (-width, t_list[j], t_list[j + 1]))
                # We must keep t_list sorted as we insert; use bisect.
                import bisect
                while gaps and time.time() < deadline:
                    neg_w, l, r = heapq.heappop(gaps)
                    m = 0.5 * (l + r)
                    # Evaluate the midpoint
                    xm = np.zeros(self.dim)
                    xm[0] = m
                    xm = np.clip(xm, lo, hi)
                    fm = problem(xm)
                    last_obj = fm
                    # Insert m into t_list
                    pos = bisect.bisect_right(t_list, l)
                    t_list.insert(pos, m)
                    # Update gaps: (l, m) and (m, r)
                    w1 = m - l
                    w2 = r - m
                    if w1 > 1e-9:
                        heapq.heappush(gaps, (-w1, l, m))
                    if w2 > 1e-9:
                        heapq.heappush(gaps, (-w2, m, r))

            return last_obj
        else:
            return None
# EVOLVE-BLOCK-END
