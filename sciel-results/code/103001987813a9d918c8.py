"""Combined optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Sobol broad exploration + adaptive (1+1)-ES + L-BFGS-B polish."""

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

        # Phase 1: Sobol exploration
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=n, scramble=True, seed=42)
            sobol.random(1)  # skip origin
            n_init = min(40, budget // 5)
            pts = sobol.random(n_init) * 10.0 - 5.0
        except Exception:
            rng_init = np.random.default_rng(42)
            n_init = min(40, budget // 5)
            pts = rng_init.random((n_init, n)) * 10.0 - 5.0

        for i in range(n_init):
            if calls >= budget or elapsed() > deadline:
                break
            safe_call(pts[i])

        # Phase 2: L-BFGS-B from best point
        if calls < budget and elapsed() < deadline:
            def obj1(x):
                nonlocal calls
                if calls >= budget or elapsed() > deadline:
                    return best_f
                f = safe_call(x)
                return f

            try:
                res = minimize(obj1, best_x, method='L-BFGS-B',
                               bounds=[(-5, 5)] * n,
                               options={'maxiter': 80, 'gtol': 1e-8, 'maxfun': budget - calls})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 3: Adaptive (1+1)-ES with restarts
        if calls < budget and elapsed() < deadline:
            sigma = 1.0
            x_curr = best_x.copy()
            f_curr = best_f
            success_count = 0
            window = 10
            stall = 0
            max_stall = 50

            while calls < budget and elapsed() < deadline:
                # Generate mutant
                z = np.random.default_rng().standard_normal(n)
                x_mut = x_curr + sigma * z
                x_mut = np.clip(x_mut, -5.0, 5.0)
                f_mut = safe_call(x_mut)

                if f_mut < f_curr:
                    x_curr = x_mut.copy()
                    f_curr = f_mut
                    success_count += 1
                    stall = 0
                else:
                    stall += 1

                # Adaptive sigma every window steps
                if success_count >= window:
                    sigma *= 1.5
                    success_count = 0
                elif success_count <= window // 3:
                    sigma *= 0.7
                    success_count = 0

                # Restart if stalled
                if stall >= max_stall:
                    # Perturb from best with larger step
                    x_curr = best_x + np.random.default_rng().standard_normal(n) * 2.0
                    x_curr = np.clip(x_curr, -5.0, 5.0)
                    f_curr = safe_call(x_curr)
                    sigma = 1.0
                    stall = 0
                    success_count = 0

                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x_curr.copy()

        # Phase 4: Final L-BFGS-B polish
        if calls < budget and elapsed() < deadline:
            def obj2(x):
                nonlocal calls
                if calls >= budget or elapsed() > deadline:
                    return best_f
                return safe_call(x)

            try:
                res = minimize(obj2, best_x, method='L-BFGS-B',
                               bounds=[(-5, 5)] * n,
                               options={'maxiter': 50, 'gtol': 1e-10, 'maxfun': budget - calls})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
