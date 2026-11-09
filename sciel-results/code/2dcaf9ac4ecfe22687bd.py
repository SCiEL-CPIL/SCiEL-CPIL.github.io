"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart optimizer: LHS sampling + Powell refinement + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        call_count = 0
        best_x = None
        best_f = np.inf

        def tracked_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Latin Hypercube sampling for diverse initial points
        n_lhs = min(50, budget // 10)  # 50 LHS points, leaving budget for refinement
        lhs_points = self._latin_hypercube(n_lhs, dim, seed=42)

        lhs_results = []
        for x in lhs_points:
            if call_count >= budget:
                break
            if time.time() > deadline - 120:  # reserve time for refinement
                break
            f = tracked_func(x)
            lhs_results.append((f, x))

        if not lhs_results:
            # Fallback: evaluate origin
            f0 = tracked_func(np.zeros(dim))
            lhs_results = [(f0, np.zeros(dim))]

        # Sort by function value
        lhs_results.sort(key=lambda t: t[0])

        # Phase 2: Powell refinement on top candidates
        n_powell = min(5, len(lhs_results), (budget - call_count) // 20)
        powell_results = []

        for i in range(n_powell):
            if call_count >= budget:
                break
            if time.time() > deadline - 60:
                break

            x0 = lhs_results[i][1]

            def objective(x, _cf=tracked_func):
                return _cf(x)

            try:
                res = minimize(objective, x0, method='Powell',
                               bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': 200, 'maxfev': 200,
                                        'xtol': 1e-8, 'ftol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                powell_results.append((res.fun, res.x))
            except Exception:
                # If Powell fails, just use the initial point
                f_init = lhs_results[i][0]
                powell_results.append((f_init, x0))

        # Phase 3: Nelder-Mead polish on the best point
        if best_x is not None and call_count < budget and time.time() < deadline - 30:
            def objective_nm(x, _cf=tracked_func):
                return _cf(x)

            try:
                res_nm = minimize(objective_nm, best_x, method='Nelder-Mead',
                                  bounds=[(-5.0, 5.0)] * dim,
                                  options={'maxiter': 500, 'maxfev': 100,
                                           'xatol': 1e-10, 'fatol': 1e-10,
                                           'adaptive': True})
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
            except Exception:
                pass

        # Phase 4: Final micro-perturbation pass to catch any remaining sub-optimality
        remaining = budget - call_count
        if remaining > 0 and best_x is not None and time.time() < deadline - 5:
            local_sigma = 0.05
            for _ in range(min(remaining, 30)):
                if call_count >= budget:
                    break
                if time.time() > deadline - 2:
                    break
                y = np.clip(best_x + local_sigma * np.random.randn(dim), -5.0, 5.0)
                fy = tracked_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    local_sigma *= 0.7
                else:
                    local_sigma *= 0.8

        if best_x is None:
            best_x = np.zeros(dim)
            if call_count < budget:
                best_f = float(tracked_func(best_x))
            else:
                best_f = float('inf')

        return best_x, best_f

    def _latin_hypercube(self, n_samples, dim, seed=42):
        """Generate n_samples points in [-5, 5]^dim using Latin Hypercube sampling."""
        rng = np.random.RandomState(seed)
        points = np.zeros((n_samples, dim))
        for d in range(dim):
            perm = rng.permutation(n_samples)
            points[:, d] = (perm + rng.random(n_samples)) / n_samples * 10.0 - 5.0
        return points
# EVOLVE-BLOCK-END
