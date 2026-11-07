import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-Start L-BFGS-B optimizer for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    1. Global Exploration: Sample points using a Halton-like sequence to cover the space.
    2. Local Exploitation: Run L-BFGS-B from the best global candidates.
    3. Budget Control: Strictly limit function calls to 500.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_f = float('inf')
        self.best_x = None
        self.func = None
        self.deadline = None

    def _counted_func(self, x):
        """Wrapper to count function calls and update best."""
        if self.call_count >= self.budget:
            return self.best_f
        x_clipped = np.clip(x, -5, 5)
        f_val = self.func(x_clipped)
        self.call_count += 1
        
        if f_val < self.best_f:
            self.best_f = f_val
            self.best_x = x_clipped.copy()
            
        return f_val

    def _halton_sequence(self, n, dim):
        """Generate n points in [0, 1]^dim using Halton sequence."""
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = []
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for i in range(1, n + 1):
            point = []
            for j in range(dim):
                point.append(halton(i, primes[j]))
            points.append(point)
        return np.array(points)

    def __call__(self, func):
        t_start = time.time()
        self.deadline = t_start + 290 # Safety margin for 300s limit
        self.func = func
        self.call_count = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)
        
        # Evaluate origin first
        f0 = self._counted_func(np.zeros(self.dim))
        
        # Phase 1: Global Search
        # Use 20 Halton points to cover the space
        n_global = 20
        halton_points = self._halton_sequence(n_global, self.dim)
        
        # Map [0, 1] to [-5, 5]
        global_points = halton_points * 10 - 5
        
        for p in global_points:
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            self._counted_func(p)
        
        # Phase 2: Local Refinement
        # We have spent ~21 calls. Remaining budget: ~479.
        # L-BFGS-B with finite differences uses (dim+1) calls per gradient step.
        # For dim=5, that's 6 calls per step.
        # We can afford roughly 479 / 6 = ~80 steps total.
        # We will run L-BFGS-B from the best point found so far.
        
        x0 = self.best_x.copy()
        remaining_calls = self.budget - self.call_count
        
        try:
            res = minimize(
                self._counted_func, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(-5, 5)] * self.dim,
                options={'maxiter': 100, 'maxfun': remaining_calls, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            pass
        
        # If we have significant budget left, try a second restart from a perturbed best point
        if self.call_count < self.budget - 50 and time.time() < self.deadline:
             x1 = self.best_x + 0.1 * np.random.standard_normal(self.dim)
             x1 = np.clip(x1, -5, 5)
             remaining_calls = self.budget - self.call_count
             try:
                 res2 = minimize(
                    self._counted_func, 
                    x1, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 20, 'maxfun': remaining_calls, 'ftol': 1e-12, 'gtol': 1e-8}
                )
             except Exception:
                 pass

        return self.best_x, self.best_f
