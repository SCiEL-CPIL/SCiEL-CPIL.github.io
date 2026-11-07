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

        # Strategy: Nelder-Mead direct search with random restarts
        # Nelder-Mead uses a simplex of dim+1 vertices and performs
        # reflect/expand/contract/shrink operations to navigate the
        # landscape. This is a derivative-free method that exploits
        # local smooth structure without modeling a covariance matrix.

        dim = self.dim
        budget = self.budget
        np.random.seed(42)

        from scipy.optimize import minimize

        # Phase 1: Global exploration with Nelder-Mead from diverse starts
        # Allocate budget across restarts; each restart gets ~60% of
        # remaining budget, but we cap maxiter to keep individual runs
        # fast so we can do more restarts.

        # Diverse starting points: origin, random interior, boundary-near
        rng = np.random.RandomState(42)
        start_points = []
        start_points.append(np.zeros(dim))
        for _ in range(4):
            start_points.append(rng.uniform(-4, 4, dim))
        for _ in range(2):
            start_points.append(rng.uniform(-5, -4, dim))
            start_points.append(rng.uniform(4, 5, dim))

        # Shuffle starts for diversity
        idx = rng.permutation(len(start_points))
        start_points = [start_points[i] for i in idx]

        no_improve_count = 0
        max_no_improve = 3

        for i, x0 in enumerate(start_points):
            if self.calls >= budget - 5:
                break
            if time.time() > self.deadline - 5:
                break

            f0 = wrapped_func(x0)
            if not np.isfinite(f0):
                continue

            f_before = self.best_f

            # Run Nelder-Mead from this start
            # maxiter controls the number of function evaluations roughly
            # (each iteration uses ~dim+1 evals for the simplex update)
            remaining = budget - self.calls - 5
            # Cap maxiter so that a single restart doesn't consume too much budget
            maxiter = max(1, min(remaining // 10, 50))

            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead',
                               options={'maxiter': maxiter,
                                        'xatol': 1e-8,
                                        'fatol': 1e-12,
                                        'initial_simplex': None})
            except Exception:
                continue

            if np.isfinite(res.fun) and res.fun < f_before:
                self.best_f = res.fun
                self.best_x = np.clip(res.x, -5, 5).copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break

        # Phase 2: Exploit the best point found with a tighter Nelder-Mead
        if self.calls < budget - 3 and time.time() < self.deadline - 3:
            remaining = budget - self.calls - 2
            if remaining > 10:
                maxiter = max(1, min(remaining // 5, 100))
                try:
                    res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                                   options={'maxiter': maxiter,
                                            'xatol': 1e-10,
                                            'fatol': 1e-14,
                                            'initial_simplex': None})
                    if np.isfinite(res.fun) and res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

        return self.best_x, self.best_f
