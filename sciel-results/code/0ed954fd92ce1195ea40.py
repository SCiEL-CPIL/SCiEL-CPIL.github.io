"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer for Rosenbrock-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 280.0  # Safety margin before 300s limit
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0

        def wrapped_func(x):
            self.calls_made += 1
            if self.calls_made > self.budget:
                raise RuntimeError("Budget exceeded")
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Use a fraction of budget for multiple restarts
        # Rosenbrock has a narrow valley; multi-start helps find the valley
        n_restarts = 20
        budget_per_restart = max(10, self.budget // n_restarts)

        for _ in range(n_restarts):
            if time.time() > self.deadline:
                break
            if self.calls_made >= self.budget:
                break

            # Start from a random point, biased towards center for Rosenbrock
            # Rosenbrock minimum is at (1,1,...,1) but with hidden shift/rotation
            # Use a mix of center and random starts
            if rng.random() < 0.5:
                x0 = rng.uniform(-2, 2, self.dim)
            else:
                x0 = rng.uniform(-5, 5, self.dim)

            # Clip to box
            x0 = np.clip(x0, -5, 5)

            # Use Nelder-Mead with limited iterations
            # maxfev controls function evaluations
            maxfev = min(budget_per_restart, self.budget - self.calls_made)
            if maxfev < 10:
                break

            try:
                result = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxfev': maxfev,
                        'xatol': 1e-4,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )

                # Update best if this run found something better
                if result.fun < self.best_f:
                    self.best_f = result.fun
                    self.best_x = result.x.copy()

            except Exception:
                # If optimization fails, continue to next restart
                continue

        # If we have time and budget left, do a final polish with more iterations
        if self.best_x is not None and self.calls_made < self.budget and time.time() < self.deadline:
            remaining = self.budget - self.calls_made
            if remaining >= 20:
                try:
                    result = scipy.optimize.minimize(
                        wrapped_func,
                        self.best_x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': remaining,
                            'xatol': 1e-5,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if result.fun < self.best_f:
                        self.best_f = result.fun
                        self.best_x = result.x.copy()
                except Exception:
                    pass

        # Ensure we return a valid result
        if self.best_x is None:
            # Fallback: return a point we've evaluated
            self.best_x = np.zeros(self.dim)
            self.best_f = wrapped_func(self.best_x)

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
