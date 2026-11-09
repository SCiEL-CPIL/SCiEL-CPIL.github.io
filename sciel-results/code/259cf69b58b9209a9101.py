# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Multi-restart Nelder-Mead and L-BFGS-B hybrid optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)
        deadline = time.time() + 290

        def counted_func(x):
            if self.calls >= self.budget or time.time() > deadline:
                raise Exception("Budget exceeded")
            self.calls += 1
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Strategy:
        # 1. A few Nelder-Mead runs to find a basin (derivative-free, robust)
        # 2. Polish with L-BFGS-B using numerical gradient (fast convergence)
        
        # Generate diverse starting points
        starts = [np.zeros(self.dim)]
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, self.dim))
            
        # Phase 1: Nelder-Mead exploration
        for x0 in starts[:5]:
            if self.calls >= self.budget - 50 or time.time() > deadline:
                break
            try:
                res = minimize(
                    counted_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-6}
                )
            except Exception:
                pass
        
        # Phase 2: L-BFGS-B polishing from the best point found so far
        # and a few other promising points
        def grad(x):
            g = np.zeros(self.dim)
            h = 1e-5
            x0 = x.copy()
            fx0 = counted_func(x0)
            for i in range(self.dim):
                x_plus = x0.copy()
                x_plus[i] += h
                fx_plus = counted_func(x_plus)
                g[i] = (fx_plus - fx0) / h
            return g

        # Use the current best and a few random perturbations of it
        polish_starts = [self.best_x]
        for _ in range(5):
            polish_starts.append(self.best_x + rng.normal(0, 0.1, self.dim))
            
        for x0 in polish_starts:
            if self.calls >= self.budget - 10 or time.time() > deadline:
                break
            try:
                res = minimize(
                    counted_func, 
                    x0, 
                    jac=grad, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 50, 'gtol': 1e-5}
                )
            except Exception:
                pass

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
