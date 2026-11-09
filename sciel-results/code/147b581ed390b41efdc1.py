"""Optimized solver for BBOB f9 (rotated Rosenbrock) using L-BFGS-B + BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hybrid L-BFGS-B and BFGS optimizer for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # State tracking
        call_count = [0]
        best_f = [np.inf]
        best_x = [None]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        def remaining_time():
            return deadline - time.time()

        def has_budget():
            return call_count[0] < budget and remaining_time() > 3.0

        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(42)

        # Generate a focused set of diverse starting points
        starts = []
        # Origin and near-origin
        starts.append(np.zeros(dim))
        for _ in range(10):
            starts.append(rng.normal(0, 1.5, dim))
        # Diagonals
        for s in [-3, -2, -1, 0, 1, 2, 3]:
            starts.append(np.full(dim, s * 0.8))
        # Random spread
        for _ in range(15):
            starts.append(rng.uniform(-4.5, 4.5, dim))
        # Axis-aligned perturbations
        for i in range(dim):
            for v in [-2.0, -1.0, 1.0, 2.0]:
                p = np.zeros(dim)
                p[i] = v
                starts.append(p)

        # Clip and deduplicate
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            if not any(np.linalg.norm(s - u) < 0.1 for u in unique_starts):
                unique_starts.append(s)

        # Evaluate starts (limited to save budget)
        max_start_evals = min(25, len(unique_starts), budget - 50)
        start_evals = []
        for s in unique_starts[:max_start_evals]:
            if not has_budget():
                break
            f = counted_func(s)
            start_evals.append((f, s.copy()))

        start_evals.sort(key=lambda t: t[0])
        top_starts = [s for _, s in start_evals[:min(8, len(start_evals))]]

        overall_best_f = best_f[0]
        overall_best_x = best_x[0].copy() if best_x[0] is not None else np.zeros(dim)

        # Phase 1: L-BFGS-B from top starts (gradient-based, fast in Rosenbrock valley)
        for i, start in enumerate(top_starts):
            if not has_budget():
                break
            if remaining_time() < 20.0:
                break

            try:
                # Use tighter tolerances for later iterations
                if i == 0:
                    maxiter = 200
                    ftol = 1e-14
                    gtol = 1e-12
                elif i == 1:
                    maxiter = 150
                    ftol = 1e-13
                    gtol = 1e-11
                else:
                    maxiter = 100
                    ftol = 1e-12
                    gtol = 1e-10

                res = minimize(
                    counted_func,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                continue

        # Phase 2: Dense BFGS with numerical Hessian (quadratic convergence in dim=5)
        if has_budget() and remaining_time() > 15.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='BFGS',
                    bounds=bounds,
                    options={'maxiter': 100, 'gtol': 1e-14, 'disp': False}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 3: Tight L-BFGS-B polish
        if has_budget() and remaining_time() > 10.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Phase 4: Final ultra-tight BFGS polish
        if has_budget() and remaining_time() > 5.0:
            try:
                res = minimize(
                    counted_func,
                    overall_best_x,
                    method='BFGS',
                    bounds=bounds,
                    options={'maxiter': 50, 'gtol': 1e-16, 'disp': False}
                )
                if res.fun < overall_best_f:
                    overall_best_f = res.fun
                    overall_best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Use tracked best if better
        if best_x[0] is not None and best_f[0] < overall_best_f:
            overall_best_f = best_f[0]
            overall_best_x = best_x[0]

        if overall_best_x is None:
            if best_x[0] is not None:
                return best_x[0], best_f[0]
            return np.zeros(dim), 0.0

        return overall_best_x, overall_best_f
