import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-Start L-BFGS-B Optimizer for Rosenbrock with adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def _valley_descent(self, func, x0, max_calls=50):
        """
        Coordinate descent along the Rosenbrock valley structure.
        The Rosenbrock valley follows x_i ≈ x_{i-1}^2.
        This method uses that structure to make efficient moves.
        """
        x = np.clip(x0.copy(), -5, 5)
        f = func(x)
        calls = 1
        self.calls_made += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        
        # Try to follow the valley: x_i should be close to x_{i-1}^2
        # We'll do a few rounds of coordinate updates with valley-aware steps
        for round in range(3):
            if calls >= max_calls or time.time() > self.deadline:
                break
            for i in range(self.dim):
                if calls >= max_calls or time.time() > self.deadline:
                    break
                
                # Valley-aware step: try to set x_i close to x_{i-1}^2 (for i>0)
                # or to 1 (for i=0, since the global min has x_0=1)
                target = 1.0 if i == 0 else x[i-1]**2
                target = np.clip(target, -5, 5)
                
                # Try moving towards the target
                trial = x.copy()
                trial[i] = target
                trial = np.clip(trial, -5, 5)
                
                f_trial = func(trial)
                calls += 1
                self.calls_made += 1
                
                if f_trial < f:
                    x = trial
                    f = f_trial
                    if f < self.best_f:
                        self.best_f = f
                        self.best_x = x.copy()
        
        return x, f

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, use 285s to be safe
        self.deadline = self.start_time + 285.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # Define starting points ordered by likelihood
        # Rosenbrock global min is at (1,1,1,1,1).
        # With shift s in [-4,4], min is at 1+s, so in [-3, 5].
        
        rng = np.random.default_rng(42)
        
        # Priority-ordered starts:
        # 1. Standard Rosenbrock min (most likely if shift is small)
        # 2. Nearby perturbations of (1,1,1,1,1)
        # 3. Random in likely region [-3, 5]
        # 4. Random in full box
        
        starts = [np.ones(self.dim)]
        
        # Small perturbations around (1,1,1,1,1) to handle small shifts
        for scale in [0.5, 1.0, 2.0]:
            starts.append(np.ones(self.dim) + rng.normal(0, scale, self.dim))
        
        # Random in likely region
        for _ in range(5):
            starts.append(rng.uniform(-3, 5, self.dim))
        
        # A few in full box for robustness
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        # Remove duplicates
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-6):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        
        # Try each start with L-BFGS-B
        for x0 in unique_starts:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 15:
                continue
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-10
                    }
                )
                
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
                    
            except StopIteration:
                break
            except Exception:
                continue
        
        # Valley-descent refinement on the best point found so far
        if self.best_x is not None and self.calls_made < self.budget and time.time() < self.deadline:
            remaining = self.budget - self.calls_made
            if remaining > 30:
                try:
                    x_valley, f_valley = self._valley_descent(func, self.best_x, max_calls=min(remaining, 40))
                    if f_valley < self.best_f:
                        self.best_f = f_valley
                        self.best_x = x_valley
                except Exception:
                    pass
        
        # Final high-precision polish if we have budget and time
        if self.best_x is not None and self.calls_made < self.budget and time.time() < self.deadline:
            remaining = self.budget - self.calls_made
            if remaining > 10:
                try:
                    res = scipy.optimize.minimize(
                        wrapped_func,
                        self.best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={
                            'maxiter': 300,
                            'ftol': 1e-14,
                            'gtol': 1e-12
                        }
                    )
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5, 5)
                except StopIteration:
                    pass
                except Exception:
                    pass

        # If no calls were made or best_x is still None
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = func(np.clip(x0, -5, 5))
        
        return self.best_x, self.best_f
