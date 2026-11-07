import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0

        def wrapped_func(x):
            self.call_count += 1
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > self.deadline:
                raise TimeoutError
            x = np.clip(np.asarray(x, dtype=float), -self.bound, self.bound)
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = np.inf
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        def has_budget(n=1):
            return (self.call_count + n <= self.budget) and (time.time() < self.deadline - 0.5)

        def polish(x0, method='Nelder-Mead'):
            if not has_budget(3):
                return
            x0 = np.clip(np.asarray(x0, dtype=float), -self.bound, self.bound)
            try:
                if method == 'Nelder-Mead':
                    res = scipy.optimize.minimize(
                        wrapped_func, x0, method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True}
                    )
                else:
                    res = scipy.optimize.minimize(
                        wrapped_func, x0, method='Powell',
                        options={'maxiter': 600, 'ftol': 1e-12, 'xtol': 1e-12}
                    )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(np.asarray(res.x, dtype=float), -self.bound, self.bound)
            except TimeoutError:
                pass
            except Exception:
                pass

        # ---------- Phase 1: Global coverage ----------
        # 1) Stratified: one random point per octant (2^dim = 32)
        for i in range(2 ** self.dim):
            if not has_budget():
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                bit = (i >> d) & 1
                if bit == 0:
                    x[d] = np.random.uniform(-self.bound, 0.0)
                else:
                    x[d] = np.random.uniform(0.0, self.bound)
            wrapped_func(x)

        # 2) Uniform random fill up to ~60% of budget for global exploration
        n_random = int(0.60 * self.budget)
        for _ in range(n_random):
            if not has_budget():
                break
            x = np.random.uniform(-self.bound, self.bound, self.dim)
            wrapped_func(x)

        if self.best_x is None:
            return np.zeros(self.dim), 0.0

        # ---------- Phase 2: Local polish from best points ----------
        # Polish the current best with Nelder-Mead (robust, no derivatives)
        polish(self.best_x, 'Nelder-Mead')

        # Re-sample a few jittered starts around the best to escape shallow local minima,
        # then polish each.
        n_jitter = 6
        for _ in range(n_jitter):
            if not has_budget(40):
                break
            jitter = np.random.uniform(-0.6, 0.6, self.dim)
            x0 = np.clip(self.best_x + jitter, -self.bound, self.bound)
            polish(x0, 'Nelder-Mead')

        # A Powell pass (good for axis-aligned structure) from the best
        if has_budget(40):
            polish(self.best_x, 'Powell')

        # ---------- Phase 3: Final exploit ----------
        # Use remaining budget for a final high-precision Nelder-Mead run
        if has_budget(20):
            polish(self.best_x, 'Nelder-Mead')

        return self.best_x, self.best_f
