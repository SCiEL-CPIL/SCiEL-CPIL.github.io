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
        deadline = start_time + 295.0

        best_x = None
        best_f = np.inf
        calls_used = 0
        candidates = []

        rng = np.random.default_rng(42)

        def check_budget():
            nonlocal calls_used
            if calls_used >= self.budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def safe_call(x):
            nonlocal calls_used, best_f, best_x, candidates
            if not check_budget():
                return None
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
            return fx

        # Phase 1: Global Exploration
        # Keep Sobol + Random, maybe adjust counts
        n_sobol = 50
        n_random = 20

        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        X_sobol = sobol_points * 10.0 - 5.0

        X_random = rng.uniform(-5.0, 5.0, size=(n_random, self.dim))

        X_initial = np.vstack([X_sobol, X_random])

        for x in X_initial:
            if not check_budget():
                break
            safe_call(x)

        if best_x is None:
            x = np.zeros(self.dim)
            safe_call(x)
            if best_x is None:
                return np.zeros(self.dim), 0.0

        # Phase 2: Local Refinement with L-BFGS-B
        candidates.sort(key=lambda t: t[0])

        top_candidates = []
        for fx, x in candidates:
            if len(top_candidates) >= 5:
                break
            is_unique = True
            for _, x_prev in top_candidates:
                if np.linalg.norm(x - x_prev) < 0.3:
                    is_unique = False
                    break
            if is_unique:
                top_candidates.append((fx, x))

        bounds = [(-5.0, 5.0)] * self.dim

        for fx0, x0 in top_candidates:
            if not check_budget() or time.time() > deadline - 15.0:
                break

            def objective(x):
                fx = safe_call(x)
                if fx is None:
                    raise Exception("Budget exceeded")
                return fx

            try:
                res = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 50
                    }
                )
            except Exception:
                pass

        # Phase 3: Built-in Powell Refinement (replaces custom Powell)
        if check_budget() and time.time() < deadline - 5.0:
            def objective_powell(x):
                fx = safe_call(x)
                if fx is None:
                    raise Exception("Budget exceeded")
                return fx

            try:
                res_powell = minimize(
                    objective_powell,
                    best_x,
                    method='Powell',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-12,
                        'xtol': 1e-8,
                        'disp': False
                    }
                )
            except Exception:
                pass

        # Phase 4: Final Nelder-Mead Polish
        if check_budget() and time.time() < deadline - 2.0:
            def objective_nm(x):
                fx = safe_call(x)
                if fx is None:
                    raise Exception("Budget exceeded")
                return fx

            try:
                res_nm = minimize(
                    objective_nm,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'fatol': 1e-14,
                        'xatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            if check_budget():
                safe_call(best_x)

        return best_x, best_f
