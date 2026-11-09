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

        # Use Powell's method, which is robust for BBOB
        # We will run it on the current best point
        if self.calls < N and np.isfinite(self.best_fx):
            try:
                # Calculate maxfev to stay within budget
                # Powell uses roughly d * (1 + d/2) calls per iteration? 
                # Safer to limit maxfev to remaining budget
                remaining = N - self.calls
                maxfev = max(10, remaining // 2) # Use half the budget for Powell
                
                res = minimize(
                    eval_func,
                    self.best_x,
                    method='Powell',
                    options={
                        'maxiter': 100, # Iterations
                        'maxfev': maxfev, # Function evals
                        'disp': False
                    }
                )
            except StopIteration:
                pass
            except Exception:
                pass
        
        # Final polish with Nelder-Mead on the best point found by Powell
        if self.calls < N and np.isfinite(self.best_fx):
            try:
                remaining = N - self.calls
                # NM uses ~d+1 calls per iteration
                maxiter = max(1, remaining // (d + 1))
                
                res = minimize(
                    eval_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': maxiter,
                        'xatol': 1e-8,
                        'fatol': 1e-12,
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
