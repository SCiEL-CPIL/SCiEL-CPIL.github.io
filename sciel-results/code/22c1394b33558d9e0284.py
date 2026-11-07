"""Model-and-Optimize with (1+1)-ES polish for Sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Model-based optimizer with (1+1)-ES for BBOB-style Sphere functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We stop at 320s.
        deadline = start_time + 320.0
        
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)
        
        N = self.dim
        bounds_low = -5.0
        bounds_high = 5.0

        def wrapped_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            x = np.clip(x, bounds_low, bounds_high)
            
            val = func(x)
            self.calls += 1
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # PHASE 1: MODEL AND OPTIMIZE
        # Estimate the center c of the Sphere f(x) = ||x-c||^2 + k
        # Using N+1 points: x0=0, xi = D*ei
        
        D = 4.0
        points = [np.zeros(N)]
        for i in range(N):
            p = np.zeros(N)
            p[i] = D
            points.append(p)
        
        values = []
        for p in points:
            try:
                v = wrapped_func(p)
                values.append(v)
            except (ValueError, TimeoutError, Exception):
                break
        
        # If model estimation failed, fallback to random search + ES
        if len(values) < N + 1:
            rng = np.random.default_rng(42)
            for _ in range(10):
                if self.calls >= self.budget: break
                x = rng.uniform(bounds_low, bounds_high, N)
                try:
                    wrapped_func(x)
                except:
                    break
            if self.calls < self.budget and time.time() < deadline:
                self._run_11es(wrapped_func, self.best_x, N)
            return (self.best_x, self.best_f)

        # Solve for c
        # f(xi) - f(x0) = ||xi||^2 - ||x0||^2 - 2 ci^T (xi - x0)
        # 2 ci^T (xi - x0) = ||xi||^2 - ||x0||^2 - (f(xi) - f(x0))
        # Since x0=0 and xi = D*ei:
        # 2 ci * D = D^2 - (f(xi) - f(0))
        # ci = (D^2 - (f(xi) - f(0))) / (2D)
        
        c_est = np.zeros(N)
        f0 = values[0]
        for i in range(N):
            fi = values[i+1]
            c_est[i] = (D**2 - (fi - f0)) / (2.0 * D)
        
        x_start = np.clip(c_est, bounds_low, bounds_high)
        
        # Evaluate the estimated center
        if not np.allclose(x_start, self.best_x):
            try:
                wrapped_func(x_start)
            except (ValueError, TimeoutError, Exception):
                pass

        # PHASE 2: POLISH WITH (1+1)-ES
        # (1+1)-ES is very efficient for smooth convex functions like Sphere
        if self.calls < self.budget and time.time() < deadline:
            self._run_11es(wrapped_func, self.best_x, N)
        
        return (self.best_x, self.best_f)

    def _run_11es(self, func, x0, dim):
        """(1+1)-ES with adaptive step size for local polish."""
        # Initial step size: small, as we expect to be close to optimum
        sigma = 0.1 
        x_best = x0.copy()
        f_best = self.best_f
        
        # Parameters
        inc_factor = 1.1
        dec_factor = 0.8
        min_sigma = 1e-12
        max_sigma = 1.0
        
        rng = np.random.default_rng(12345)
        
        while self.calls < self.budget and time.time() < self.deadline - 2:
            # Generate offspring
            z = rng.standard_normal(dim)
            x_new = x_best + sigma * z
            x_new = np.clip(x_new, -5.0, 5.0)
            
            try:
                f_new = func(x_new)
            except (ValueError, TimeoutError):
                break
            except Exception:
                break
            
            if f_new < f_best:
                x_best = x_new
                f_best = f_new
                self.best_f = f_best
                self.best_x = x_best.copy()
                sigma *= inc_factor
            else:
                sigma *= dec_factor
            
            sigma = np.clip(sigma, min_sigma, max_sigma)
            
            # Early stop if sigma is very small and no improvement for a while
            # (Optional, but budget check handles it)
        
        return (self.best_x, self.best_f)

# EVOLVE-BLOCK-END
