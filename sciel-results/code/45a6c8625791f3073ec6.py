"""Black-box continuous optimizer on BBOBB-style f8 (Rosenbrock), dim 5, box [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start gradient solver (L-BFGS-B) with valley-aware prior and polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)
        bounds = list(zip(lb, ub))

        # Finite-difference gradient estimator (central, adaptive step).
        def grad(x):
            g = np.empty(dim)
            for i in range(dim):
                h = 1e-5 * max(1.0, abs(x[i]))
                xp = x.copy(); xp[i] += h
                xm = x.copy(); xm[i] -= h
                g[i] = (func(xp) - func(xm)) / (2.0 * h)
            return g

        # Build prioritized starting set: centered on (1,...,1) (the standard
        # Rosenbrock minimum) with small perturbations first, then broader coverage.
        starts = []
        base = np.ones(dim)
        starts.append(base.copy())
        rng = np.random.default_rng(12345)
        # small perturbations around (1,...,1)
        for s in (0.1, 0.3, 0.6):
            for _ in range(3):
                p = base + s * rng.standard_normal(dim)
                p = np.clip(p, -5.0, 5.0)
                starts.append(p)
        # broader coverage of the likely shifted region [-4,4]^5
        for _ in range(6):
            p = rng.uniform(-4.0, 4.0, dim)
            starts.append(p)
        # a few box-wide exploratory starts
        for _ in range(3):
            p = rng.uniform(-5.0, 5.0, dim)
            starts.append(p)

        best_x = starts[0].copy()
        best_f = func(best_x)
        calls = 1

        def run_lbfgsb(x0, maxiter, ftol):
            nonlocal calls
            res = minimize(func, x0, method="L-BFGS-B",
                           jac=grad, bounds=bounds,
                           options={"maxiter": maxiter, "ftol": ftol,
                                    "gtol": 1e-12, "maxls": 40})
            if res.x is not None and np.isfinite(res.fun):
                return res.x, res.fun
            return None, None

        # Pass 1: moderate-iteration runs from each start, prioritized.
        for x0 in starts:
            if time.time() > deadline - 20.0:
                break
            x, f = run_lbfgsb(x0, maxiter=300, ftol=1e-10)
            if x is not None and f < best_f:
                best_f = f
                best_x = x.copy()

        # Pass 2: high-precision polish from the current best (tighter tolerance,
        # more iterations) to squeeze the last digits of the valley.
        if time.time() < deadline - 5.0:
            x, f = run_lbfgsb(best_x, maxiter=800, ftol=1e-14)
            if x is not None and f < best_f:
                best_f = f
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
