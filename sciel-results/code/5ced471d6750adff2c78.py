"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start Powell Optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        # 300s limit, use 280s to be safe.
        self.time_limit = 280.0 
        
        d = self.dim
        N = self.budget
        
        self.best_x = np.zeros(d)
        self.best_fx = float('inf')
        
        # Wrapper to count calls, enforce limits, and track best
        def eval_func(x):
            if self.calls >= N:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            fx = func(x)
            if fx < self.best_fx:
                self.best_fx = fx
                self.best_x = x.copy()
            return fx

        # Initialize with a few random points to find a decent start
        rng = np.random.default_rng(42)
        
        # Evaluate a few random points to initialize best_x
        for _ in range(5):
            x0 = rng.uniform(-4.5, 4.5, d)
            try:
                eval_func(x0)
            except StopIteration:
                break

        # If we still don't have a valid best (e.g. all calls failed?), try 0
        if not np.isfinite(self.best_fx):
            try:
                eval_func(np.zeros(d))
            except StopIteration:
                pass

        # Strategy: Use L-BFGS-B which uses a quasi-Newton (BFGS) approximation.
        # This is better for anisotropic landscapes like Bent Cigar than coordinate descent (Powell)
        # because it adapts the search direction to the local curvature.
        
        if self.calls < N and np.isfinite(self.best_fx):
            try:
                remaining = N - self.calls
                # L-BFGS-B uses ~2*dim+1 calls per iteration for numerical gradients.
                # We allow it to use most of the remaining budget.
                maxfev = remaining
                
                res = minimize(
                    eval_func,
                    self.best_x,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 1000,
                        'maxfev': maxfev,
                        'ftol': 1e-15,
                        'gtol': 1e-8,
                        'disp': False
                    }
                )
            except StopIteration:
                pass
            except Exception:
                pass
        
        # Final polish with Nelder-Mead if budget remains
        # NM can sometimes refine the solution if L-BFGS-B stopped early due to gradient tolerance
        if self.calls < N and np.isfinite(self.best_fx):
            try:
                remaining = N - self.calls
                if remaining > 10:
                    maxiter = max(1, remaining // (d + 1))
                    
                    res = minimize(
                        eval_func,
                        self.best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': maxiter,
                            'xatol': 1e-10,
                            'fatol': 1e-15,
                            'disp': False
                        }
                    )
            except StopIteration:
                pass
            except Exception:
                pass

        # Ensure we return a valid point
        if not np.isfinite(self.best_fx):
            self.best_fx = 0.0
            self.best_x = np.zeros(d)

        return self.best_x, self.best_fx
# EVOLVE-BLOCK-END
