import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s

        dim = self.dim
        budget = self.budget
        eval_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            eval_count += 1
            fx = func(x)
            if not np.isfinite(fx):
                fx = 1e30
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol quasi-random sampling for global exploration
        # 64 points for good early coverage
        n_sobol = 64
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sobol.random(n_sobol) * 10.0 - 5.0

        # Store all evaluated points and values for ridge estimation
        eval_points = []
        eval_values = []

        for pt in sobol_points:
            if eval_count >= budget or time.time() > deadline:
                break
            fx = wrapped_func(pt)
            eval_points.append(pt.copy())
            eval_values.append(fx)

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Estimate the principal ridge direction from the best points
        # Sort by function value and take the top few
        sorted_indices = np.argsort(eval_values)
        n_top = min(8, len(eval_points))
        top_points = np.array([eval_points[i] for i in sorted_indices[:n_top]])
        top_values = np.array([eval_values[i] for i in sorted_indices[:n_top]])

        # Estimate ridge direction via PCA on top points, weighted by inverse value
        # Points with lower values are closer to the ridge
        # Use a simple approach: direction from worst of top to best
        if n_top >= 2:
            # Weighted centroid
            weights = 1.0 / (top_values - top_values[0] + 1e-6)
            weights /= weights.sum()
            centroid = np.average(top_points, axis=0, weights=weights)
            
            # Direction: from the point farthest from centroid (among top) to centroid
            # Actually, for a ridge, we want the direction along which values change slowly
            # Let's use the direction between the best and the 2nd best
            dir_vec = top_points[1] - top_points[0]
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-6:
                dir_vec = np.ones(dim) / np.sqrt(dim)
            else:
                dir_vec = dir_vec / norm
        else:
            dir_vec = np.ones(dim) / np.sqrt(dim)

        # Phase 3: 1-D line search along the estimated ridge direction
        # Use scipy's minimize_scalar for 1-D optimization
        if eval_count < budget - 10 and time.time() < deadline - 5:
            try:
                def ridge_func(t):
                    x = best_x + t * dir_vec
                    x = np.clip(x, -5.0, 5.0)
                    return wrapped_func(x)

                # Search in a reasonable range
                res = minimize_scalar(
                    ridge_func,
                    bounds=(-5.0, 5.0),
                    method='bounded',
                    options={'maxiter': 50, 'xatol': 1e-6}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(best_x + res.x * dir_vec, -5.0, 5.0)
            except Exception:
                pass

        # Phase 4: Local refinement with BFGS from the best point
        if eval_count < budget - 20 and time.time() < deadline - 10:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='BFGS',
                    options={'maxiter': 50, 'gtol': 1e-8, 'disp': False}
                )
                if res.fun < best_f and np.isfinite(res.fun):
                    best_x = np.clip(res.x, -5.0, 5.0)
                    best_f = res.fun
            except Exception:
                pass

        # Phase 5: If budget remains, do a few more local refinements from
        # perturbed versions of the best point to diversify
        if eval_count < budget - 20 and time.time() < deadline - 10:
            rng = np.random.default_rng(123)
            for _ in range(3):
                if eval_count >= budget - 10 or time.time() > deadline - 5:
                    break
                x0 = best_x + rng.normal(0, 0.5, dim)
                x0 = np.clip(x0, -5.0, 5.0)
                try:
                    res = minimize(
                        wrapped_func,
                        x0,
                        method='BFGS',
                        options={'maxiter': 30, 'gtol': 1e-8, 'disp': False}
                    )
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_x = np.clip(res.x, -5.0, 5.0)
                        best_f = res.fun
                except Exception:
                    pass

        # Phase 6: Final polish with Nelder-Mead if budget remains
        if eval_count < budget - 10 and time.time() < deadline - 5:
            try:
                remaining = budget - eval_count - 5
                if remaining > 10:
                    res2 = minimize(
                        wrapped_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'maxfev': remaining, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = np.clip(res2.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
