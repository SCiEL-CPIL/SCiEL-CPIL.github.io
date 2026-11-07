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
        self.deadline = time.time() + 280.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0
        rng = np.random.default_rng(seed=42)

        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x = np.clip(x, -self.bound, self.bound)
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Global Search with Random Sampling
        # Increase samples to improve basin discovery probability in 5D
        n_global_samples = 150
        for _ in range(n_global_samples):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = rng.uniform(-self.bound, self.bound, self.dim)
            wrapped_func(p)

        if self.best_x is None:
            return np.zeros(self.dim), 0.0

        # Phase 2: Local Optimization with Powell
        try:
            res = scipy.optimize.minimize(
                wrapped_func,
                self.best_x,
                method='Powell',
                options={
                    'maxiter': 80,
                    'ftol': 1e-14,
                    'xtol': 1e-14
                }
            )
            if res.fun < self.best_f:
                self.best_f = res.fun
                self.best_x = res.x.copy()
        except TimeoutError:
            pass
        except RuntimeError:
            pass
        except Exception:
            pass

        # Phase 3: High-precision polish with Nelder-Mead
        if self.call_count < self.budget - 20 and time.time() < self.deadline - 5:
            try:
                res2 = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-14,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if res2.fun < self.best_f:
                    self.best_f = res2.fun
                    self.best_x = res2.x.copy()
            except TimeoutError:
                pass
            except RuntimeError:
                pass
            except Exception:
                pass

        return self.best_x, self.best_f
