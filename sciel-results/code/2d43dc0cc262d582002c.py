"""Combined optimizer: adaptive multi-phase SLSQP with Nelder-Mead fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Adaptive multi-phase SLSQP with Nelder-Mead fallback for 5D Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        best_x = None
        best_f = np.inf
        call_count = [0]
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_left():
            return deadline - time.time()

        def can_continue():
            return call_count[0] < budget - 5 and time_left() > 5.0

        # Phase 1: Light random sampling to find initial good region
        n_initial = min(15, max(8, budget // 30))
        for i in range(n_initial):
            if not can_continue():
                break
            x0 = rng.uniform(-5.0, 5.0, dim)
            f0 = wrapped_func(x0)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        # Phase 2: Primary SLSQP from best point
        remaining = budget - call_count[0] - 20
        if remaining > 10 and can_continue():
            max_iter = max(10, min(80, remaining // (dim + 1)))
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 3: Local perturbations around best with decreasing scale
        n_local = min(10, max(4, (budget - call_count[0] - 30) // 35))
        for j in range(n_local):
            if not can_continue():
                break

            scale = 2.0 / (j + 1)
            x0 = best_x + rng.normal(0, scale, dim)
            x0 = np.clip(x0, -5.0, 5.0)
            f0 = wrapped_func(x0)

            remaining = budget - call_count[0] - 15
            if remaining > 10 and can_continue():
                max_iter = max(5, min(40, remaining // (dim + 1)))
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
                    )
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0)
                except Exception:
                    remaining = budget - call_count[0] - 10
                    if remaining > 5 and can_continue():
                        try:
                            res = minimize(
                                wrapped_func,
                                x0,
                                method='Nelder-Mead',
                                options={
                                    'maxiter': min(50, remaining // 3),
                                    'xatol': 1e-10,
                                    'fatol': 1e-12,
                                    'adaptive': True,
                                    'disp': False
                                }
                            )
                            if np.isfinite(res.fun) and res.fun < best_f:
                                best_f = res.fun
                                best_x = np.clip(res.x, -5.0, 5.0)
                        except Exception:
                            pass

        # Phase 4: Global random restarts
        n_global = min(5, max(2, (budget - call_count[0] - 40) // 40))
        for j in range(n_global):
            if not can_continue():
                break

            x0 = rng.uniform(-5.0, 5.0, dim)
            f0 = wrapped_func(x0)

            remaining = budget - call_count[0] - 15
            if remaining > 10 and can_continue():
                max_iter = max(10, min(50, remaining // (dim + 1)))
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
                    )
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0)
                except Exception:
                    pass

        # Phase 5: Final high-precision SLSQP polish
        if best_x is not None and can_continue():
            remaining = budget - call_count[0] - 5
            if remaining > 5:
                max_iter_final = max(10, min(100, remaining // (dim + 1)))
                try:
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter_final, 'ftol': 1e-15, 'disp': False}
                    )
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0)
                except Exception:
                    remaining = budget - call_count[0] - 3
                    if remaining > 3 and can_continue():
                        try:
                            res = minimize(
                                wrapped_func,
                                best_x,
                                method='Nelder-Mead',
                                options={
                                    'maxiter': min(50, remaining // 2),
                                    'xatol': 1e-12,
                                    'fatol': 1e-15,
                                    'adaptive': True,
                                    'disp': False
                                }
                            )
                            if np.isfinite(res.fun) and res.fun < best_f:
                                best_f = res.fun
                                best_x = np.clip(res.x, -5.0, 5.0)
                        except Exception:
                            pass

        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = wrapped_func(best_x)
            else:
                best_f = np.inf

        return best_x, best_f
# EVOLVE-BLOCK-END
