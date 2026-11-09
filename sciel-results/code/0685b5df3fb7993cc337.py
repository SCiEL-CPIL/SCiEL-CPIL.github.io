import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # Wrap func to count calls and check time
        def wrapped_func(x):
            self.calls += 1
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise ValueError("Time limit exceeded")
            return func(x)

        # Initial point
        # Bent Cigar is unimodal, but shifted. 
        # Starting at 0 is a reasonable guess if the shift is not too large.
        # The problem states minimizer is in [-4, 4]^5.
        x0 = np.zeros(self.dim)
        
        # Phase 1: Powell Method
        # Powell is derivative-free and handles anisotropy well.
        try:
            res = scipy.optimize.minimize(
                wrapped_func, 
                x0, 
                method='Powell',
                options={'maxiter': 50, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            x_best = res.x
            f_best = res.fun
        except Exception:
            # If Powell fails or exceeds budget/time, fallback to whatever best we have
            # Or just stop. But we need to return something.
            # If it threw exception due to budget/time, we should use the last known best.
            # However, scipy.optimize.minimize doesn't easily expose "last best" on exception.
            # So we must be careful. 
            # Better to implement a manual loop or use a method that is less likely to throw.
            # Actually, if it throws, we might have consumed all budget.
            # Let's assume it completes or we handle the exception by returning the initial point if no calls were made?
            # No, the task says "exception later keeps the calls made".
            # So if it throws, the evaluator sees the calls made so far.
            # We should try to return the best x we found.
            # Since we don't have a global best tracker in this simple wrapper, 
            # let's modify the wrapper to store the best x and f.
            pass

        # Let's refine the wrapper to track best
        self.calls = 0
        self.best_x = x0
        self.best_f = wrapped_func(x0)

        def wrapped_func(x):
            self.calls += 1
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise ValueError("Time limit exceeded")
            f_val = func(x)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            return f_val

        # Phase 1: Powell
        try:
            res = scipy.optimize.minimize(
                wrapped_func, 
                self.best_x, 
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-12, 'ftol': 1e-12}
            )
        except Exception:
            # Continue with best found so far
            pass

        # Phase 2: Polish with Nelder-Mead or Coordinate Descent
        # If we have budget left, use it to polish.
        if self.calls < self.budget and time.time() < deadline:
            # Nelder-Mead is good for final polishing
            try:
                res2 = scipy.optimize.minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

        return self.best_x, self.best_f
