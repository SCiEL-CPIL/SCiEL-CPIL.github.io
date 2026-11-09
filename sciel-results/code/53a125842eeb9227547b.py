import numpy as np
import scipy.optimize
import time

class Optimizer:
    """
    Multi-start L-BFGS-B optimizer for rotated Rosenbrock (f9).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        self.eval_count = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = float('inf')

        class BudgetExceeded(Exception):
            pass

        def wrapped_func(x):
            if time.time() - self.start_time > 295:
                raise BudgetExceeded()
            if self.eval_count >= self.budget:
                raise BudgetExceeded()
            x = np.clip(x, -5, 5)
            self.eval_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = 1e10
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Quick exploration to get early AOCC credit
        # Evaluate a small set of diverse points directly
        rng = np.random.default_rng(42)
        
        # Quick scan: center + a few spread points
        quick_starts = [np.zeros(self.dim)]
        for i in range(6):
            quick_starts.append(rng.uniform(-3, 3, self.dim))
        
        for x0 in quick_starts:
            if self.eval_count >= self.budget or time.time() - self.start_time > 295:
                break
            try:
                wrapped_func(x0)
            except BudgetExceeded:
                break

        # Phase 2: Multi-start L-BFGS-B with diverse starting points
        bounds = [(-5, 5)] * self.dim
        
        # Generate starting points: mix of center, random, and structured
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Best found so far (if it improved)
        if self.best_f < 1e10:
            starts.append(self.best_x.copy())
        
        # Random points in [-4, 4] - well-spread
        for _ in range(12):
            starts.append(rng.uniform(-4, 4, self.dim))
        
        # Structured points: axis-aligned and diagonal patterns
        for axis in range(self.dim):
            for val in [-2, 2]:
                p = np.zeros(self.dim)
                p[axis] = val
                starts.append(p)
        
        # Diagonal patterns
        for scale in [1.0, 2.0, 3.0]:
            starts.append(np.full(self.dim, scale))
            starts.append(np.full(self.dim, -scale))
        
        # Shuffle to mix quality of starts
        rng.shuffle(starts)
        
        # Track how many successful L-BFGS-B runs we've done
        successful_runs = 0
        
        for x0 in starts:
            remaining = self.budget - self.eval_count
            if remaining < 15:
                break
            if time.time() - self.start_time > 293:
                break
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 150,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 30
                    }
                )
                successful_runs += 1
            except BudgetExceeded:
                break
            except Exception:
                continue
        
        # Phase 3: High-precision polish on the best point found
        if self.eval_count < self.budget - 10 and time.time() - self.start_time < 293:
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-16,
                        'gtol': 1e-12,
                        'maxls': 50
                    }
                )
            except BudgetExceeded:
                pass
            except Exception:
                pass
        
        # Phase 4: If budget remains, try a few more local refinements
        if self.eval_count < self.budget - 10 and time.time() - self.start_time < 293:
            # Try small perturbations around best point
            for _ in range(5):
                if self.eval_count >= self.budget - 5:
                    break
                if time.time() - self.start_time > 293:
                    break
                perturbation = rng.normal(0, 0.1, self.dim)
                x_try = np.clip(self.best_x + perturbation, -5, 5)
                try:
                    wrapped_func(x_try)
                except BudgetExceeded:
                    break
                except Exception:
                    continue
        
        return self.best_x, self.best_f
