"""Model-and-Optimize with high-precision coordinate polish for Sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Model-based optimizer with high-precision coordinate polish for BBOB-style Sphere functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
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

        # If model estimation failed, fallback to random search + polish
        if len(values) < N + 1:
            rng = np.random.default_rng(42)
            for _ in range(10):
                if self.calls >= self.budget:
                    break
                x = rng.uniform(bounds_low, bounds_high, N)
                try:
                    wrapped_func(x)
                except:
                    break
            if self.calls < self.budget and time.time() < deadline:
                self._run_polish(wrapped_func, self.best_x, N)
            return (self.best_x, self.best_f)

        # Solve for c
        # f(xi) - f(x0) = ||xi||^2 - ||x0||^2 - 2 ci^T (xi - x0)
        # Since x0=0 and xi = D*ei:
        # 2 ci * D = D^2 - (f(xi) - f(0))
        # ci = (D^2 - (f(xi) - f(0))) / (2D)

        c_est = np.zeros(N)
        f0 = values[0]
        for i in range(N):
            fi = values[i + 1]
            c_est[i] = (D**2 - (fi - f0)) / (2.0 * D)

        x_start = np.clip(c_est, bounds_low, bounds_high)

        # Evaluate the estimated center
        try:
            wrapped_func(x_start)
        except (ValueError, TimeoutError, Exception):
            pass

        # PHASE 2: HIGH-PRECISION POLISH
        # Use coordinate-wise ternary search to refine the minimum
        # This is extremely effective for separable convex functions like Sphere
        if self.calls < self.budget and time.time() < deadline:
            self._run_polish(wrapped_func, self.best_x, N)

        return (self.best_x, self.best_f)

    def _run_polish(self, func, x0, dim):
        """High-precision coordinate-wise polish using ternary search."""
        x_best = x0.copy()
        f_best = self.best_f
        
        # We use a small window around the current best point
        # Since the model-based estimate is likely very close, we start with a small window
        # and expand if needed, but for Sphere, the minimum is local and sharp.
        
        # Number of iterations for polish
        max_iter = 50
        # Initial window size for ternary search
        window = 0.1
        
        for _ in range(max_iter):
            if self.calls >= self.budget or time.time() > self.deadline - 2:
                break
            
            x_current = x_best.copy()
            f_current = f_best
            improved = False
            
            for i in range(dim):
                if self.calls >= self.budget or time.time() > self.deadline - 2:
                    break
                
                # Ternary search along dimension i
                # We search in [x_current[i] - window, x_current[i] + window]
                a = x_current[i] - window
                b = x_current[i] + window
                
                # Clip to bounds
                a = max(a, -5.0)
                b = min(b, 5.0)
                
                if b - a < 1e-12:
                    continue
                
                # Perform ternary search
                # We do a fixed number of steps to ensure precision
                steps = 20 
                for _ in range(steps):
                    if self.calls >= self.budget or time.time() > self.deadline - 2:
                        break
                    
                    m1 = a + (b - a) / 3.0
                    m2 = b - (b - a) / 3.0
                    
                    x1 = x_current.copy()
                    x1[i] = m1
                    x2 = x_current.copy()
                    x2[i] = m2
                    
                    try:
                        f1 = func(x1)
                        f2 = func(x2)
                    except (ValueError, TimeoutError, Exception):
                        break
                    
                    if f1 < f2:
                        b = m2
                    else:
                        a = m1
                        
                # After ternary search, the best point is near (a+b)/2
                x_opt_i = (a + b) / 2.0
                x_try = x_current.copy()
                x_try[i] = x_opt_i
                
                try:
                    f_try = func(x_try)
                except (ValueError, TimeoutError, Exception):
                    break
                
                if f_try < f_current:
                    x_current = x_try
                    f_current = f_try
                    improved = True
            
            # Update best
            if f_current < f_best:
                x_best = x_current.copy()
                f_best = f_current
                self.best_f = f_best
                self.best_x = x_best.copy()
            
            # Shrink window if no improvement
            if not improved:
                window *= 0.5
                if window < 1e-10:
                    break
            else:
                # If improved, keep window or shrink slightly to converge
                window *= 0.8

        return (self.best_x, self.best_f)

# EVOLVE-BLOCK-END
