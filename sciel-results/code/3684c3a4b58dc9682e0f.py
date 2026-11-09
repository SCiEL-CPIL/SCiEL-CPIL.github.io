"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Deterministic Sobol sampling followed by Nelder-Mead on top-10 HV contributors."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Phase 1: Deterministic Sobol sequence sampling
        n_sobol = 140
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_samples = sampler.random(n_sobol)
        X_sobol = lo + sobol_samples * span
        X_sobol = np.clip(X_sobol, lo, hi)

        # Evaluate Sobol samples
        F_sobol = np.array([problem(x) for x in X_sobol])
        used = n_sobol

        X_all = X_sobol.copy()
        F_all = F_sobol.copy()

        ref_point = np.array([3000.0, 0.06])

        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume for points strictly dominated by ref (minimization)."""
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            max_f2 = -np.inf
            for i in range(len(pts)):
                f1, f2 = pts[i]
                if f2 < max_f2:
                    hv += (ref[0] - f1) * (max_f2 - f2)
                else:
                    max_f2 = f2
                    hv += (ref[0] - f1) * (ref[1] - f2)
            return hv

        def get_pareto_front(points):
            """Return indices of non-dominated points."""
            n = len(points)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if np.all(points[i] <= points[j]) and np.any(points[i] < points[j]):
                        dominated[j] = True
                    elif np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        def hv_contribution(idx, points, ref):
            """Compute the hypervolume contribution of point idx to the set."""
            full_hv = hypervolume_2d(points, ref)
            remaining = np.delete(points, idx, axis=0)
            if len(remaining) == 0:
                return full_hv
            reduced_hv = hypervolume_2d(remaining, ref)
            return full_hv - reduced_hv

        # Phase 2: Identify top-10 hypervolume contributors from Pareto front
        pareto_idx = get_pareto_front(F_all)
        if len(pareto_idx) == 0:
            pareto_idx = np.array([0])

        pareto_points = F_all[pareto_idx]
        n_pareto = len(pareto_idx)

        # Compute contributions for all Pareto front points
        contributions = np.zeros(n_pareto)
        for k, idx in enumerate(pareto_idx):
            contributions[k] = hv_contribution(idx, F_all, ref_point)

        # Select top-10 contributors
        n_top = min(10, n_pareto)
        top_local_idx = np.argsort(contributions)[::-1][:n_top]
        top_indices = pareto_idx[top_local_idx]

        remaining_budget = budget - used

        if remaining_budget > 0:
            # Create diverse weightings spanning the trade-off
            if n_pareto >= 2:
                f1_min = np.min(pareto_points[:, 0])
                f1_max = np.max(pareto_points[:, 0])
                f2_min = np.min(pareto_points[:, 1])
                f2_max = np.max(pareto_points[:, 1])

                f1_range = f1_max - f1_min if f1_max > f1_min else 1.0
                f2_range = f2_max - f2_min if f2_max > f2_min else 1.0

                n_weights = n_top
                weights = []
                for k in range(n_weights):
                    if n_weights == 1:
                        w1 = 0.5
                    else:
                        w1 = k / (n_weights - 1)
                    if w1 < 1e-6:
                        weights.append(np.array([0.0, 1.0]))
                    elif 1 - w1 < 1e-6:
                        weights.append(np.array([1.0, 0.0]))
                    else:
                        # Normalize by ranges to balance objectives
                        w_norm = np.array([w1 / f1_range, (1 - w1) / f2_range])
                        w_norm = w_norm / (np.sum(w_norm) + 1e-10)
                        weights.append(w_norm)
            else:
                weights = [np.array([0.5, 0.5])]

            # Allocate remaining budget among top-10 runs
            n_runs = n_top
            evals_per_run = max(5, remaining_budget // n_runs)

            # Ensure total doesn't exceed remaining budget
            total_planned = evals_per_run * n_runs
            if total_planned > remaining_budget:
                evals_per_run = max(3, remaining_budget // n_runs)

            for i, idx in enumerate(top_indices):
                if used >= budget - 1:
                    break

                x0 = X_all[idx].copy()
                w = weights[i % len(weights)]
                maxfev = evals_per_run

                if maxfev < 3:
                    maxfev = 3

                def objective(x, w=w):
                    nonlocal used
                    if used >= budget:
                        return 0.0
                    fx = problem(x)
                    used += 1
                    return np.dot(fx, w)

                options = {
                    'maxiter': maxfev,
                    'maxfev': maxfev,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'disp': False
                }

                try:
                    result = minimize(objective, x0, method='Nelder-Mead',
                                      bounds=list(zip(lo, hi)), options=options)
                    if result.fun is not None and used < budget:
                        x_best = np.clip(result.x, lo, hi)
                        f_best = problem(x_best)
                        used += 1
                        X_all = np.vstack([X_all, x_best])
                        F_all = np.vstack([F_all, f_best])
                except Exception:
                    pass

        return X_all
# EVOLVE-BLOCK-END
