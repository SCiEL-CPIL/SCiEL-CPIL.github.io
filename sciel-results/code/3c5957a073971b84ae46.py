import numpy as np
import scipy.optimize as so
import time

class Optimizer:
    """Optimizer using structured initialization and L-BFGS-B / Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        self.deadline = 320.0

    def _check_time(self):
        if time.time() > self.deadline:
            raise TimeoutError("Time limit exceeded")

    def _eval(self, func, x):
        self._check_time()
        x = np.clip(x, -5.0, 5.0)
        self.calls += 1
        if self.calls > self.budget:
            raise ValueError("Budget exceeded")
        
        fx = func(x)
        
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return fx

    def __call__(self, func):
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None

        # Phase 1: Initial Exploration
        candidates = []
        rng = np.random.default_rng(42)
        
        candidates.append(np.zeros(self.dim))
        candidates.append(np.ones(self.dim))
        candidates.append(-np.ones(self.dim))
        
        for _ in range(10):
            candidates.append(rng.choice([-1.0, 1.0], size=self.dim))
            
        for _ in range(30):
            candidates.append(rng.uniform(-5.0, 5.0, self.dim))
            
        for _ in range(20):
            candidates.append(rng.uniform(-2.0, 2.0, self.dim))

        for x in candidates:
            try:
                self._eval(func, x)
            except (TimeoutError, ValueError):
                break

        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = func(self.best_x)

        # Phase 2: Local Optimization with L-BFGS-B
        remaining_calls = self.budget - self.calls
        if remaining_calls > 0:
            def scipy_func(x):
                return self._eval(func, x)
            
            bounds = [(-5.0, 5.0)] * self.dim
            
            try:
                res = so.minimize(scipy_func, self.best_x, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 100, 'maxfun': remaining_calls})
                if res.success:
                    res_x_clipped = np.clip(res.x, -5.0, 5.0)
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = res_x_clipped
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Phase 3: Polish with Nelder-Mead
        remaining_calls = self.budget - self.calls
        if remaining_calls > 20:
            def scipy_func_nm(x):
                return self._eval(func, x)
            
            try:
                res_nm = so.minimize(scipy_func_nm, self.best_x, method='Nelder-Mead',
                                     options={'maxiter': 50, 'maxfev': remaining_calls, 'xatol': 1e-8, 'fatol': 1e-8})
                if res_nm.success:
                    res_x_clipped = np.clip(res_nm.x, -5.0, 5.0)
                    if res_nm.fun < self.best_f:
                        self.best_f = res_nm.fun
                        self.best_x = res_x_clipped
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        return self.best_x, self.best_f
