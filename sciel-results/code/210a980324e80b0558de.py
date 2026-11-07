"""Combined optimizer: multi-start SLSQP with Nelder-Mead fallback and adaptive refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start SLSQP with Nelder-Mead fallback and adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        best_x = None
        best_f = np.inf
        call_count = [0]

        def eval_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_left():
            return deadline - time.time()

        def budget_left():
            return budget - call_count[0]

        rng = np.random.default_rng(42)

        # Generate diverse starting points
        starts = []

        # Origin
        starts.append(np.zeros(dim))

        # Points near origin with small perturbations
        for _ in range(3):
            p = rng.uniform(-0.5, 0.5, dim)
            starts.append(p)

        # Random uniform points
        for _ in range(12):
            starts.append(rng.uniform(-5, 5, dim))

        # Points along each axis
        for d in range(dim):
            for val in [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]:
                p = np.zeros(dim)
                p[d] = val
                starts.append(p)

        # Diagonal points
        for val in [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]:
            starts.append(np.full(dim, val))

        # Points with two non-zero coordinates
        for _ in range(8):
            p = np.zeros(dim)
            idx = rng.choice(dim, size=2, replace=False)
            p[idx[0]] = rng.uniform(-4, 4)
            p[idx[1]] = rng.uniform(-4, 4)
            starts.append(p)

        # Structured points that might be closer to typical Rosenbrock minima
        for _ in range(5):
            p = rng.uniform(-4, 4, dim)
            # Bias toward having some coordinates near 1 (common Rosenbrock feature)
            if rng.random() < 0.5:
                p[rng.integers(0, dim)] = rng.uniform(0.5, 1.5)
            starts.append(p)

        # Shuffle starts
        indices = rng.permutation(len(starts))
        starts = [starts[i] for i in indices]

        # Phase 1: Multi-start with SLSQP, fallback to Nelder-Mead
        for start in starts:
            if call_count[0] >= budget or time_left() < 5.0:
                break

            # Try SLSQP first
            try:
                result = minimize(
                    eval_func,
                    start,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'disp': False}
                )
            except Exception:
                result = None

            # If SLSQP failed or didn't improve, try Nelder-Mead
            if result is None or not np.isfinite(result.fun) or result.fun >= best_f - 1e-12:
                try:
                    nm_result = minimize(
                        eval_func,
                        start,
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                except Exception:
                    nm_result = None
            else:
                nm_result = None

            # Update best
            if result is not None and np.isfinite(result.fun) and result.fun < best_f:
                best_f = result.fun
                best_x = np.clip(result.x, -5.0, 5.0)
            if nm_result is not None and np.isfinite(nm_result.fun) and nm_result.fun < best_f:
                best_f = nm_result.fun
                best_x = np.clip(nm_result.x, -5.0, 5.0)

            # If we've found a very good solution, break early
            if best_f < 1e-12:
                break

            # If we've used most of our budget, stop
            if call_count[0] > budget * 0.75:
                break

        # Phase 2: Local refinement around the best point found
        if best_x is not None and call_count[0] < budget and time_left() > 10.0:
            # Try SLSQP with tighter tolerances
            try:
                result = minimize(
                    eval_func,
                    best_x,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-18, 'disp': False}
                )
                if np.isfinite(result.fun) and result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass

            # Try Nelder-Mead with very tight tolerances
            if call_count[0] < budget and time_left() > 5.0:
                try:
                    nm_result = minimize(
                        eval_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
                    )
                    if np.isfinite(nm_result.fun) and nm_result.fun < best_f:
                        best_f = nm_result.fun
                        best_x = np.clip(nm_result.x, -5.0, 5.0)
                except Exception:
                    pass

            # Final SLSQP polish if budget allows
            if call_count[0] < budget and time_left() > 2.0 and best_f > 1e-15:
                try:
                    result = minimize(
                        eval_func,
                        best_x,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-20, 'disp': False}
                    )
                    if np.isfinite(result.fun) and result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0)
                except Exception:
                    pass

        # Fallback if no valid result
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = float(func(best_x))
                call_count[0] += 1
            else:
                best_f = np.inf

        return best_x, best_f
# EVOLVE-BLOCK-END
