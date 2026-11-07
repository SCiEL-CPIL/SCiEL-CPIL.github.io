import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.deadline = None

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return 1e10  # Return large value to signal termination to optimizer
        if time.time() > self.deadline:
            return 1e10
        self.calls += 1
        val = self.func(x)
        if val < self.best_f:
            self.best_f = val
            self.best_x = np.array(x)
        return val

    def __call__(self, func):
        self.func = func
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None
        self.start_time = time.time()
        self.deadline = self.start_time + 280.0  # 20s safety margin

        d = self.dim
        rng = np.random.default_rng(42)

        # Phase 1: Initial random sampling to find a good starting point
        # Bent Cigar has a narrow valley, so random points might be far.
        # We sample a few points to get a baseline.
        n_init = 10
        init_points = []
        for _ in range(n_init):
            if self.calls >= self.budget - 50: # Reserve budget for optimization
                break
            x = rng.uniform(-5, 5, size=d)
            f = self._evaluate(x)
            if self.best_x is not None:
                init_points.append((f, x))
        
        if self.best_x is None:
            # Fallback if even init fails (should not happen)
            self.best_x = np.zeros(d)
            self._evaluate(self.best_x)

        # Phase 2: Local optimization using L-BFGS-B
        # We will try to optimize from the best initial point found.
        # L-BFGS-B is effective for anisotropic problems.
        
        # Sort init points by function value
        if init_points:
            init_points.sort(key=lambda item: item[0])
            best_start = init_points[0][1]
        else:
            best_start = self.best_x

        # Run L-BFGS-B
        # maxiter is set high, but we rely on budget/time guard.
        # L-BFGS-B stops when it converges or maxiter is reached.
        
        try:
            res = minimize(
                self._evaluate, 
                best_start, 
                method='L-BFGS-B',
                bounds=[(-5, 5)] * d,
                options={
                    'maxiter': 200, # Should be enough for 5D
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
        except Exception:
            pass

        # If we have significant budget left, we might do another restart
        # to ensure we haven't missed the global minimum (though BBOB f12 is unimodal).
        # For unimodal functions, L-BFGS-B from a random start is very robust.
        
        # Check if we should do a second optimization from a different random point
        # just in case the first one got stuck in a local minimum (unlikely for f12).
        if self.calls < self.budget - 100 and time.time() < self.deadline - 30:
            # Generate a new random start
            x_new = rng.uniform(-5, 5, size=d)
            # Evaluate it
            f_new = self._evaluate(x_new)
            
            if f_new < self.best_f:
                self.best_x = x_new
                self.best_f = f_new
                
            try:
                res2 = minimize(
                    self._evaluate, 
                    x_new, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * d,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
            except Exception:
                pass

        if self.best_x is None:
            self.best_x = np.zeros(d)
            self.best_f = 0.0

        return self.best_x, self.best_f
