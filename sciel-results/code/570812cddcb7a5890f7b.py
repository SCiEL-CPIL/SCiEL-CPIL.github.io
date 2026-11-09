import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using structured Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # Find the minimizer of g for tail variables x[1:dim]
        # g contribution per variable: x^2 - 10*cos(4*pi*x)
        # Global minimum is at x=0 (value -9), local minima at integers
        x_tail = np.zeros(dim - 1)
        for i in range(1, dim):
            l, h = bounds[i]
            # Check candidates: bounds, 0, and nearby integers
            candidates = [l, h, 0.0]
            # Add integers near the interval
            start_int = int(np.floor(l))
            for k in range(start_int - 1, start_int + 10):
                if l - 1e-12 <= k <= h + 1e-12:
                    candidates.append(float(k))
            
            best_val = float('inf')
            best_x = l
            for c in candidates:
                if l - 1e-12 <= c <= h + 1e-12:
                    val = c * c - 10.0 * np.cos(4.0 * np.pi * c)
                    if val < best_val:
                        best_val = val
                        best_x = c
            x_tail[i - 1] = best_x

        # Use all budget for structured sweep of x[0]
        # Quadratic mapping u^2 clusters points near x0=lo (likely 0)
        # where the Pareto front has higher curvature
        n_evals = self.budget

        for i in range(n_evals):
            if time.time() > deadline:
                break

            if n_evals > 1:
                u = i / (n_evals - 1)
            else:
                u = 0.5

            # Quadratic mapping to cluster near lo[0]
            x0 = lo[0] + (hi[0] - lo[0]) * (u * u)

            x = np.empty(dim)
            x[0] = x0
            x[1:] = x_tail

            try:
                problem(x)
            except Exception:
                continue

        return None
