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
        self.deadline = time.time() + 285  # 15s margin before 300s kill
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

        dim = self.dim
        budget = self.budget

        # Strategy: Multi-start Nelder-Mead + coordinate descent
        # Nelder-Mead for local optimization, followed by coordinate descent for fine-tuning.
        # Fewer starts with more budget per start improves convergence.

        # Generate diverse starting points (reduced count for more budget per start)
        np.random.seed(42)
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(5):
            starts.append(np.random.uniform(-3, 3, dim))
        starts.append(np.random.uniform(-5, -3, dim))
        starts.append(np.random.uniform(3, 5, dim))

        num_starts = len(starts)

        from scipy.optimize import minimize

        for i in range(num_starts):
            if self.calls >= budget - 10:
                break
            if time.time() > self.deadline - 10:
                break

            x0 = starts[i].copy()
            f0 = wrapped_func(x0)

            if not np.isfinite(f0):
                continue

            # Phase 1: Nelder-Mead local optimization
            if self.calls < budget - 20 and time.time() < self.deadline - 10:
                try:
                    remaining = min(60, budget - self.calls - 10)
                    if remaining > 20:
                        maxiter = max(1, remaining // 5)
                        res = minimize(wrapped_func, x0, method='Nelder-Mead',
                                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10,
                                                'initial_simplex': None})
                        if np.isfinite(res.fun) and res.fun < self.best_f:
                            self.best_f = res.fun
                            self.best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

            # Phase 2: Coordinate descent for fine-tuning
            if self.calls < budget - 10 and time.time() < self.deadline - 5:
                try:
                    x = self.best_x.copy()
                    f = self.best_f
                    step = 0.5
                    for coord in range(dim):
                        if self.calls >= budget - 5:
                            break
                        if time.time() > self.deadline - 3:
                            break
                        for _ in range(20):
                            if self.calls >= budget - 5:
                                break
                            if time.time() > self.deadline - 3:
                                break
                            x_new = x.copy()
                            x_new[coord] = x[coord] + step
                            f_new = wrapped_func(x_new)
                            if np.isfinite(f_new) and f_new < f:
                                x = x_new
                                f = f_new
                            else:
                                x_new = x.copy()
                                x_new[coord] = x[coord] - step
                                f_new = wrapped_func(x_new)
                                if np.isfinite(f_new) and f_new < f:
                                    x = x_new
                                    f = f_new
                                else:
                                    step *= 0.5
                                    if step < 1e-4:
                                        break
                        if step < 1e-4:
                            break
                    if np.isfinite(f) and f < self.best_f:
                        self.best_f = f
                        self.best_x = np.clip(x, -5, 5).copy()
                except Exception:
                    pass

        # Final polish with tight Nelder-Mead around the best point
        if self.calls < budget - 5 and time.time() < self.deadline - 5:
            try:
                remaining = min(30, budget - self.calls - 3)
                if remaining > 15:
                    maxiter = max(1, remaining // 4)
                    res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                                   options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12})
                    if np.isfinite(res.fun) and res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        return self.best_x, self.best_f
