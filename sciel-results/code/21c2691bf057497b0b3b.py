import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 320.0

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()

        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(np.clip(x, -5.0, 5.0))

        n = self.dim
        best_x = np.zeros(n)
        best_fx = float('inf')

        rng = np.random.default_rng(123)
        starts = [np.zeros(n)]
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, n))

        for x0 in starts:
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.time_limit:
                break
            remaining = self.budget - self.calls
            if remaining < 5:
                break
            try:
                max_iter = min(200, remaining // (n + 1))
                if max_iter < 1:
                    break
                res = minimize(
                    eval_func,
                    x0,
                    method='Powell',
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'xatol': 1e-8}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except StopIteration:
                break
            except Exception:
                continue

        remaining = self.budget - self.calls
        if remaining > 10 and best_x is not None and time.time() - self.start_time < self.time_limit - 5:
            try:
                max_iter = min(50, remaining // (n + 1))
                if max_iter >= 1:
                    res = minimize(
                        eval_func,
                        best_x,
                        method='Powell',
                        options={'maxiter': max_iter, 'ftol': 1e-14, 'xatol': 1e-12}
                    )
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = res.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(n)
            if self.calls < self.budget:
                best_fx = eval_func(best_x)
            else:
                best_fx = 0.0

        return best_x, best_fx
