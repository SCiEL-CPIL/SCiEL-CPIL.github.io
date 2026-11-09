import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-Start L-BFGS-B Optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, use 280s
        self.deadline = self.start_time + 280.0

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

        # Define starting points
        # Rosenbrock global min is at (1,1,1,1,1).
        # With shift s in [-4,4], min is at 1+s.
        # Likely region for min: [-3, 5].
        
        starts = []
        # 1. Standard Rosenbrock min
        starts.append(np.ones(self.dim))
        # 2. Origin
        starts.append(np.zeros(self.dim))
        # 3. Random in likely region
        rng = np.random.default_rng(42)
        for _ in range(4):
            starts.append(rng.uniform(-3, 4, self.dim))
        # 4. Random in full box
        for _ in range(2):
            starts.append(rng.uniform(-5, 5, self.dim))

        # Remove duplicates roughly
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        
        # Phase 1: Coarse optimization with L-BFGS-B
        for x0 in unique_starts:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            # Check if x0 is within bounds
            x0 = np.clip(x0, -5, 5)
            
            # If we have very little budget left, skip complex optimizers
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 20:
                continue

            try:
                # L-BFGS-B is good for bounded optimization
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-6,
                        'gtol': 1e-4
                    }
                )
                
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
                    
            except StopIteration:
                break
            except Exception:
                continue

        # Phase 2: High-precision polish on the best solution
        if self.best_x is not None and self.calls_made < self.budget:
            if time.time() < self.deadline:
                try:
                    # Use a tighter tolerance and more iterations for polishing
                    res_polish = scipy.optimize.minimize(
                        wrapped_func, 
                        self.best_x, 
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={
                            'maxiter': 200,
                            'ftol': 1e-12,
                            'gtol': 1e-10
                        }
                    )
                    
                    if res_polish.fun < self.best_f:
                        self.best_f = res_polish.fun
                        self.best_x = np.clip(res_polish.x, -5, 5)
                        
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
