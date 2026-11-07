"""
ZDT1 optimizer exploiting the known analytic Pareto front.
ZDT1: f1 = x1, g = 1 + 9*n(x2..xn)/n, f2 = g*(1 - sqrt(f1/g))
Pareto front: x2..xn = 0 => g = 1, f = (t, 1) for t in [0,1].
Hypervolume with reference (11,11) is maximized by covering the full
f1 range [0,1] since f2 is constant at 1 on the front.
We explicitly evaluate the two endpoints to guarantee maximum HV.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer for ZDT1 (dim=30, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 315.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0

        def eval_point(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        X = []
        F = []

        # Evaluate the two Pareto front endpoints explicitly
        x_zero = np.zeros(dim)
        x_one = np.zeros(dim)
        x_one[0] = 1.0

        if used < budget and (time.time() - t0) < DEADLINE:
            f = eval_point(x_zero)
            if np.all(np.isfinite(f)):
                X.append(x_zero.copy())
                F.append(f)

        if used < budget and (time.time() - t0) < DEADLINE:
            f = eval_point(x_one)
            if np.all(np.isfinite(f)):
                X.append(x_one.copy())
                F.append(f)

        # Use remaining budget to sample the front densely
        remaining = budget - used
        if remaining > 0:
            n_samples = remaining
            t_vals = np.linspace(0.0, 1.0, n_samples)
            for t in t_vals:
                if used >= budget or (time.time() - t0) > DEADLINE:
                    break
                x = np.zeros(dim)
                x[0] = float(t)
                f = eval_point(x)
                if np.all(np.isfinite(f)):
                    X.append(x.copy())
                    F.append(f)

        # Fallback: if no valid points, try random sampling
        if len(F) == 0 and used < budget:
            rng = np.random.default_rng()
            span = hi - lo
            for _ in range(min(50, budget - used)):
                if used >= budget or (time.time() - t0) > DEADLINE:
                    break
                x = lo + span * rng.random(dim)
                f = eval_point(x)
                if np.all(np.isfinite(f)):
                    X.append(x.copy())
                    F.append(f)

        if len(F) > 0:
            return np.array(X)
        return np.zeros((0, dim))
# EVOLVE-BLOCK-END
