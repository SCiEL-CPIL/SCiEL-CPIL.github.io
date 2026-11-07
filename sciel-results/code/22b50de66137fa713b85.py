import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = None
        self.time_limit = 285.0  # Safety margin from 300s limit

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            self.calls += 1
            if self.calls > self.budget:
                return self.best_f
            if time.time() - self.start_time > self.time_limit:
                return self.best_f

            x = np.asarray(x, dtype=float)
            try:
                fx = func(x)
            except Exception:
                return self.best_f

            if np.isnan(fx) or np.isinf(fx):
                return self.best_f

            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Dense random sampling to identify basins
        num_samples = 25
        candidates = []
        for _ in range(num_samples):
            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break
            x0 = rng.uniform(-5, 5, self.dim)
            fx = wrapped_func(x0)
            candidates.append((fx, x0))

        if not candidates:
            return self.best_x, self.best_f

        # Sort by fitness
        candidates.sort(key=lambda x: x[0])

        # Phase 2: Multi-start L-BFGS-B with varied tolerances
        # Use top 6 candidates for restarts
        num_restarts = 6
        top_candidates = candidates[:min(num_restarts, len(candidates))]

        remaining_budget = self.budget - self.calls
        if remaining_budget <= 0:
            return self.best_x, self.best_f

        budget_per_restart = max(10, remaining_budget // num_restarts)

        for i, (fx, x0) in enumerate(top_candidates):
            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break

            # First pass: standard L-BFGS-B
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B',
                               bounds=[(-5, 5)] * self.dim,
                               options={'maxiter': 150, 'maxfun': budget_per_restart,
                                        'ftol': 1e-10, 'gtol': 1e-6})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass

            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break

            # Second pass: tighter tolerance polish from the result
            if self.best_x is not None and self.calls < self.budget - 5:
                try:
                    res2 = minimize(wrapped_func, self.best_x, method='L-BFGS-B',
                                    bounds=[(-5, 5)] * self.dim,
                                    options={'maxiter': 100, 'maxfun': min(50, self.budget - self.calls),
                                             'ftol': 1e-14, 'gtol': 1e-10})
                    if res2.fun < self.best_f:
                        self.best_f = res2.fun
                        self.best_x = res2.x.copy()
                except Exception:
                    pass

            if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                break

        # Phase 3: If we still have budget, do a few more random restarts from perturbed best
        if self.calls < self.budget - 10 and self.best_x is not None:
            for _ in range(3):
                if self.calls >= self.budget or time.time() - self.start_time > self.time_limit:
                    break
                x_perturb = self.best_x + rng.normal(0, 0.5, self.dim)
                x_perturb = np.clip(x_perturb, -5, 5)
                fx_p = wrapped_func(x_perturb)

                if self.calls >= self.budget - 10 or time.time() - self.start_time > self.time_limit:
                    break

                try:
                    res = minimize(wrapped_func, x_perturb, method='L-BFGS-B',
                                   bounds=[(-5, 5)] * self.dim,
                                   options={'maxiter': 80, 'maxfun': min(40, self.budget - self.calls),
                                            'ftol': 1e-10, 'gtol': 1e-6})
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = res.x.copy()
                except Exception:
                    pass

        return self.best_x, self.best_f
