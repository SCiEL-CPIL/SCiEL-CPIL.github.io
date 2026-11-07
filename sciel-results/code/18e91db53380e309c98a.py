"""New program: Multi-start Nelder-Mead local optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls_made = 0
        self.budget_exceeded = False
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        def counted_func(x):
            nonlocal best_x, best_f
            if self.budget_exceeded:
                return 0.0
            self.calls_made += 1
            if self.calls_made > self.budget:
                self.budget_exceeded = True
                raise Exception("Budget Exceeded")
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Initialize with origin
        try:
            counted_func(best_x)
        except Exception:
            pass

        rng = np.random.default_rng(42)

        # Multi-start strategy
        for i in range(10):
            if self.budget_exceeded or self.calls_made >= self.budget:
                break

            # Random start point
            if i % 2 == 0:
                x_start = rng.uniform(-5, 5, size=self.dim)
            else:
                x_start = rng.normal(0, 2, size=self.dim)
                x_start = np.clip(x_start, -5, 5)

            try:
                res = minimize(counted_func, x_start, method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            except Exception:
                pass

        # Final polish if budget remains
        if not self.budget_exceeded and self.calls_made < self.budget:
            try:
                res = minimize(counted_func, best_x, method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
