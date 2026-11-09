"""Combined optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Sobol exploration + Nelder-Mead + L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget

        def elapsed():
            return time.time() - t0

        calls = 0
        best_f = np.inf
        best_x = np.zeros(n)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or elapsed() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return best_f
            if not np.isfinite(f):
                return best_f
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Sobol exploration with diverse initial points
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=n, scramble=True, seed=42)
            sobol.random(1)  # skip origin
            n_init = min(50, budget // 4)
            pts = sobol.random(n_init) * 10.0 - 5.0
        except Exception:
            rng_init = np.random.default_rng(42)
            n_init = min(50, budget // 4)
            pts = rng_init.random((n_init, n)) * 10.0 - 5.0

        for i in range(n_init):
            if calls >= budget or elapsed() > deadline:
                break
            safe_call(pts[i])

        # Phase 2: Nelder-Mead from best point (adapts to ridge geometry)
        if calls < budget and elapsed() < deadline:
            def obj_nm(x):
                nonlocal calls
                if calls >= budget or elapsed() > deadline:
                    return best_f
                return safe_call(x)

            try:
                x0 = best_x.copy()
                # Create initial simplex with small perturbation
                delta = 0.5
                initial_simplex = [x0 + delta * np.eye(n)[i] for i in range(n)]
                initial_simplex.append(x0)
                
                res = minimize(obj_nm, x0, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8,
                                        'initial_simplex': initial_simplex})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 3: L-BFGS-B polish from best point
        if calls < budget and elapsed() < deadline:
            def obj_lbfgs(x):
                nonlocal calls
                if calls >= budget or elapsed() > deadline:
                    return best_f
                return safe_call(x)

            try:
                res = minimize(obj_lbfgs, best_x, method='L-BFGS-B',
                               bounds=[(-5, 5)] * n,
                               options={'maxiter': 100, 'gtol': 1e-10, 'maxfun': budget - calls})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 4: Additional local searches from other promising points
        if calls < budget and elapsed() < deadline:
            # Find top 3 distinct points from initial exploration
            top_points = []
            seen = set()
            for i in range(n_init):
                key = tuple(np.round(pts[i], 2))
                if key not in seen:
                    seen.add(key)
                    top_points.append(pts[i])
                if len(top_points) >= 3:
                    break

            for pt in top_points:
                if calls >= budget or elapsed() > deadline:
                    break
                if np.allclose(pt, best_x, atol=0.1):
                    continue
                try:
                    res = minimize(obj_lbfgs, pt, method='L-BFGS-B',
                                   bounds=[(-5, 5)] * n,
                                   options={'maxiter': 50, 'gtol': 1e-10, 'maxfun': max(1, (budget - calls) // 2)})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 5: Final fine-tuning with adaptive (1+1)-ES
        if calls < budget and elapsed() < deadline:
            sigma = 0.1
            x_curr = best_x.copy()
            f_curr = best_f
            max_iter = min(100, budget - calls)
            
            for _ in range(max_iter):
                if calls >= budget or elapsed() > deadline:
                    break
                z = np.random.default_rng().standard_normal(n)
                x_mut = x_curr + sigma * z
                x_mut = np.clip(x_mut, -5.0, 5.0)
                f_mut = safe_call(x_mut)

                if f_mut < f_curr:
                    x_curr = x_mut.copy()
                    f_curr = f_mut
                    sigma *= 1.1
                else:
                    sigma *= 0.9

                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x_curr.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
