"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer combining random sampling, weighted-sum L-BFGS-B, and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Track all evaluated points and their objective values
        X_all = []
        F_all = []
        used = 0

        def counting_problem(x):
            nonlocal used
            if used >= budget:
                raise RuntimeError("Budget exceeded")
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            X_all.append(x.copy())
            F_all.append(f.copy())
            used += 1
            return f

        def time_check():
            return time.time() < deadline and used < budget

        # Phase 1: Random initialization
        n_init = min(15, budget // 10)
        for _ in range(n_init):
            if not time_check():
                break
            x = lo + (hi - lo) * rng.random(dim)
            counting_problem(x)

        # Phase 2: Weighted-sum decomposition with L-BFGS-B
        n_weights = 8
        weights = []
        for _ in range(n_weights):
            w = rng.dirichlet(np.ones(n_obj))
            weights.append(w)
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)
        weights.append(np.ones(n_obj) / n_obj)

        bounds_scipy = list(zip(lo, hi))

        for w in weights:
            if not time_check():
                break

            def weighted_obj(x, w=w):
                x = np.clip(x, lo, hi)
                f = counting_problem(x)
                return float(np.dot(w, f))

            if len(X_all) > 0:
                idx = rng.integers(len(X_all))
                x0 = X_all[idx].copy()
            else:
                x0 = lo + (hi - lo) * rng.random(dim)

            try:
                result = minimize(
                    weighted_obj, x0, method='L-BFGS-B',
                    bounds=bounds_scipy,
                    options={'maxiter': 10, 'maxfun': 10, 'ftol': 1e-5}
                )
            except Exception:
                pass

        # Phase 3: Local refinement around non-dominated points
        if len(F_all) > 0 and time_check():
            F_arr = np.array(F_all)
            X_arr = np.array(X_all)

            def is_dominated(f, F):
                return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

            non_dom_idx = [i for i in range(len(F_arr)) if not is_dominated(F_arr[i], F_arr)]

            for idx in non_dom_idx:
                if not time_check():
                    break
                x_center = X_arr[idx]
                for _ in range(4):
                    if not time_check():
                        break
                    step_size = 0.05 * (hi - lo)
                    x_pert = x_center + step_size * rng.standard_normal(dim)
                    counting_problem(x_pert)

        return X_all
# EVOLVE-BLOCK-END
