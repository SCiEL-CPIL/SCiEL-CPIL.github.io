import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.calls_made = 0
        self.max_time = 290.0

    def __call__(self, func):
        self.start_time = time.time()
        self.calls_made = 0
        self.best_x = None
        self.best_f = float('inf')

        def safe_func(x):
            if time.time() - self.start_time > self.max_time:
                return float('inf')
            if self.calls_made >= self.budget:
                return float('inf')
            self.calls_made += 1
            fx = float(func(x))
            if not np.isfinite(fx):
                return 1e10
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = np.array(x, copy=True)
            return fx

        def nm_objective(x):
            """Objective for scipy minimize with budget tracking."""
            if time.time() - self.start_time > self.max_time:
                raise RuntimeError("time limit")
            if self.calls_made >= self.budget:
                raise RuntimeError("budget exhausted")
            self.calls_made += 1
            fx = float(func(x))
            if not np.isfinite(fx):
                return 1e10
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = np.array(x, copy=True)
            return fx

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = []
        # Structured starts
        starts.append(np.zeros(self.dim))
        starts.append(np.ones(self.dim) * 3.0)
        starts.append(-np.ones(self.dim) * 3.0)
        starts.append(np.ones(self.dim) * 1.0)
        starts.append(-np.ones(self.dim) * 1.0)
        # Random starts in [-4, 4]
        for _ in range(8):
            starts.append(rng.uniform(-4, 4, self.dim))

        n_starts = len(starts)
        # Allocate budget: 75% for multi-start, 25% for final polish
        search_budget = int(self.budget * 0.75)
        polish_budget = self.budget - search_budget
        calls_per_start = max(15, search_budget // n_starts)

        # Run Nelder-Mead from each start
        for x0 in starts:
            if time.time() - self.start_time > self.max_time:
                break
            if self.calls_made >= search_budget:
                break

            remaining = min(calls_per_start, search_budget - self.calls_made)
            if remaining < 10:
                break

            try:
                result = minimize(
                    nm_objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except (RuntimeError, Exception):
                pass

            if result.x is not None and np.isfinite(result.fun):
                if result.fun < self.best_f:
                    self.best_f = result.fun
                    self.best_x = np.array(result.x, copy=True)

        # Final polish with tighter settings on the best point found
        if self.best_x is not None and self.calls_made < self.budget:
            remaining = self.budget - self.calls_made
            if remaining > 10:
                try:
                    result = minimize(
                        nm_objective,
                        self.best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if result.x is not None and np.isfinite(result.fun):
                        if result.fun < self.best_f:
                            self.best_f = result.fun
                            self.best_x = np.array(result.x, copy=True)
                except (RuntimeError, Exception):
                    pass

        return self.best_x, self.best_f
