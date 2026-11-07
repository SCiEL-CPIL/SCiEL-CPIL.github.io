import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(12345)
        call_count = 0
        best_x = None
        best_f = np.inf
        self._best_x = None
        self._best_f = np.inf

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 2:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            x_arr = np.asarray(x, dtype=float)
            val = float(func(x_arr))
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = np.array(x_arr, dtype=float)
                self._best_x = best_x.copy()
                self._best_f = best_f
            return val

        # Phase 1: Coarse grid + random sampling to locate the basin
        # Rosenbrock in 5D has a long narrow valley; we need to get into it
        n_grid = 5  # per dimension, 5^5 = 3125 too many, so use sparse
        # Use a sparse grid: 3 points per dim = 243, plus random
        grid_vals = np.linspace(-4, 4, 3)  # -4, 0, 4
        grid_pts = np.array(np.meshgrid(*([grid_vals] * self.dim))).T.reshape(-1, self.dim)
        
        n_random = 50
        random_pts = rng.uniform(-4, 4, (n_random, self.dim))
        
        # Add some structured points along axes
        axis_pts = []
        for i in range(self.dim):
            for v in [-3, -2, -1, 0, 1, 2, 3]:
                p = np.zeros(self.dim)
                p[i] = v
                axis_pts.append(p)
        axis_pts = np.array(axis_pts)
        
        all_pts = np.vstack([grid_pts, random_pts, axis_pts])
        
        for p in all_pts:
            if call_count >= self.budget or time.time() > deadline - 10:
                break
            try:
                safe_func(p)
            except ValueError:
                break
            except Exception:
                continue

        if best_x is None:
            return np.zeros(self.dim), 0.0

        # Phase 2: Powell's method from best point - aligns with Rosenbrock valley
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 10:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': 500, 'maxfev': remaining, 'ftol': 1e-12, 'xtol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.array(res.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # Phase 3: Multi-start with Powell from perturbed points
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 10:
            for scale in [0.5, 1.0, 2.0]:
                if remaining < 10 or time.time() > deadline - 5:
                    break
                try:
                    perturb = rng.normal(0, scale, self.dim)
                    start2 = np.clip(best_x + perturb, -5, 5)
                    res2 = minimize(
                        safe_func,
                        start2,
                        method='Powell',
                        options={'maxiter': 200, 'maxfev': remaining, 'ftol': 1e-12, 'xtol': 1e-10}
                    )
                    if res2.fun < best_f:
                        best_f = float(res2.fun)
                        best_x = np.array(res2.x, dtype=float)
                        self._best_x = best_x.copy()
                        self._best_f = best_f
                    remaining = self.budget - call_count
                except Exception:
                    break

        # Phase 4: L-BFGS-B polish from Powell result
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 5:
            try:
                res_lbfgs = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 500, 'maxfun': remaining, 'maxls': 50, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                if res_lbfgs.fun < best_f:
                    best_f = float(res_lbfgs.fun)
                    best_x = np.array(res_lbfgs.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # Phase 5: Final high-precision L-BFGS-B polish with tighter tolerances
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 3:
            try:
                res_final = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 1000, 'maxfun': remaining, 'maxls': 100, 
                             'gtol': 1e-15, 'ftol': 1e-16, 'disp': False}
                )
                if res_final.fun < best_f:
                    best_f = float(res_final.fun)
                    best_x = np.array(res_final.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        return best_x, best_f
