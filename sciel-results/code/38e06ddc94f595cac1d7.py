import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 285
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        def wrapped_func(x):
            if self.calls >= self.budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # Multi-start Nelder-Mead with more diverse starts and a final polish
        x0_list = [
            np.zeros(self.dim),
            np.random.uniform(-3, 3, self.dim),
            np.random.uniform(-4, 4, self.dim),
            np.random.uniform(-5, 5, self.dim),
        ]

        # Budget allocation: give more to earlier starts, save some for polish
        budgets = [150, 100, 80, 50]

        for i, x0 in enumerate(x0_list):
            if self.calls >= self.budget:
                break
            if time.time() > self.deadline - 5:
                break

            # Nelder-Mead in 5D: roughly 1 call per iteration for reflection,
            # but often 2-4 per iteration. Use maxiter = budget // 3 to be safe.
            maxiter = max(1, budgets[i] // 3)

            try:
                from scipy.optimize import minimize
                res = minimize(wrapped_func, x0, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-8})
            except Exception:
                continue

            if self.best_f > res.fun:
                self.best_f = res.fun
                self.best_x = res.x.copy()

        # Final polish: high-precision Nelder-Mead around the best point found
        if self.calls < self.budget and time.time() < self.deadline - 10:
            try:
                from scipy.optimize import minimize
                polish_budget = self.budget - self.calls
                maxiter = max(1, polish_budget // 3)
                res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-10,
                                        'initial_simplex': self._make_simplex(self.best_x)})
                if self.best_f > res.fun:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass

        return self.best_x, self.best_f

    def _make_simplex(self, x0):
        """Create a small simplex around x0 for the polish pass."""
        simplex = [x0.copy()]
        for i in range(self.dim):
            v = x0.copy()
            step = 0.01
            v[i] += step
            simplex.append(v)
        return np.array(simplex)
