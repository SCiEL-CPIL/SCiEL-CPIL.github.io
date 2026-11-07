"""Model-and-optimize with exact quadratic polish for Sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Model-based optimizer with exact quadratic polish for BBOB-style Sphere functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = 0.0

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 285  # Safety margin for 300s limit
        
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)
        
        N = self.dim
        bounds_low = -5.0
        bounds_high = 5.0

        def wrapped_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > self.deadline:
                raise TimeoutError("Time limit exceeded")
            
            x = np.clip(x, bounds_low, bounds_high)
            
            val = func(x)
            self.calls += 1
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # MODEL-AND-OPTIMIZE STRATEGY
        # Estimate x* using N+1 points to solve linear system
        D = 3.0
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
        
        # Fallback if model estimation fails
        if len(values) < N + 1:
            rng = np.random.default_rng(42)
            for _ in range(20):
                if self.calls >= self.budget: break
                x = rng.uniform(bounds_low, bounds_high, N)
                try:
                    wrapped_func(x)
                except:
                    break
            return (self.best_x, self.best_f)

        # Solve for x*
        A = np.zeros((N, N))
        b = np.zeros(N)
        
        x0 = points[0]
        f0 = values[0]
        
        for i in range(1, N + 1):
            xi = points[i]
            fi = values[i]
            
            diff = xi - x0
            norm_sq_diff = np.dot(xi, xi) - np.dot(x0, x0)
            val_diff = fi - f0
            
            A[i-1] = 2.0 * diff
            b[i-1] = norm_sq_diff - val_diff
        
        try:
            x_est = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x_est = np.zeros(N)
        
        x_start = np.clip(x_est, bounds_low, bounds_high)
        
        # Evaluate start point if not already done
        if not np.allclose(x_start, self.best_x, atol=1e-12):
            try:
                wrapped_func(x_start)
            except (ValueError, TimeoutError, Exception):
                pass

        # EXACT QUADRATIC POLISH PHASE
        # For f(x) = ||x - x*||^2 + C, the gradient is 2(x - x*)
        # So x* = x - grad(x)/2
        # We can estimate grad(x) using finite differences and jump to the optimum
        
        if self.calls < self.budget and time.time() < self.deadline:
            try:
                self._quadratic_polish(wrapped_func, self.best_x, N)
            except Exception:
                pass
        
        # If we have budget left, do a high-precision local search to squeeze out remaining error
        # The quadratic polish should get us very close, but floating point or slight model errors
        # might leave us at a suboptimal point.
        if self.calls < self.budget and time.time() < self.deadline:
            try:
                self._high_precision_polish(wrapped_func, self.best_x, N)
            except Exception:
                pass

        return (self.best_x, self.best_f)

    def _quadratic_polish(self, func, x0, dim):
        """Polish using exact quadratic structure: estimate gradient and jump to optimum."""
        x = x0.copy()
        h = 1e-6  # Step size for gradient estimation
        
        for iteration in range(20):
            if self.calls >= self.budget or time.time() > self.deadline - 5:
                break
            
            # Estimate gradient using central differences
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                try:
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                except (ValueError, TimeoutError):
                    break
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            # Check if we ran out of budget during gradient estimation
            if self.calls >= self.budget:
                break
            
            # For quadratic f(x) = ||x - x*||^2 + C, grad(x) = 2(x - x*)
            # So x* = x - grad(x)/2
            x_new = x - grad / 2.0
            
            # Clip to bounds
            x_new = np.clip(x_new, -5.0, 5.0)
            
            # Evaluate the new point
            try:
                f_new = func(x_new)
            except (ValueError, TimeoutError):
                break
            
            # If we made progress, move there
            if f_new < self.best_f:
                self.best_f = f_new
                self.best_x = x_new.copy()
                x = x_new
            else:
                # No progress, try a smaller step or break
                if np.linalg.norm(x_new - x) < 1e-10:
                    break
                # Try half step
                x_half = x - grad / 4.0
                x_half = np.clip(x_half, -5.0, 5.0)
                try:
                    f_half = func(x_half)
                    if f_half < self.best_f:
                        self.best_f = f_half
                        self.best_x = x_half.copy()
                        x = x_half
                    else:
                        break
                except (ValueError, TimeoutError):
                    break
        
        return (self.best_x, self.best_f)

    def _high_precision_polish(self, func, x0, dim):
        """
        Final high-precision polish using coordinate descent with shrinking steps.
        This handles any residual error from the quadratic jump or floating point noise.
        """
        x = x0.copy()
        f_curr = self.best_f
        
        # Start with very small steps, shrink as we go
        # Since we are already very close, large steps are unnecessary and wasteful
        steps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        
        for step in steps:
            if self.calls >= self.budget or time.time() > self.deadline - 1:
                break
            
            improved = True
            while improved and self.calls < self.budget and time.time() < self.deadline - 1:
                improved = False
                for i in range(dim):
                    if self.calls >= self.budget or time.time() > self.deadline - 1:
                        break
                    
                    for direction in [1.0, -1.0]:
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        
                        if np.allclose(x_trial, x, atol=1e-12):
                            continue
                        
                        try:
                            f_trial = func(x_trial)
                        except (ValueError, TimeoutError):
                            break
                        
                        if f_trial < f_curr:
                            x = x_trial
                            f_curr = f_trial
                            self.best_f = f_curr
                            self.best_x = x.copy()
                            improved = True
                    
                    if time.time() > self.deadline - 1:
                        break
            
            if self.calls >= self.budget:
                break

        return (self.best_x, self.best_f)
# EVOLVE-BLOCK-END
