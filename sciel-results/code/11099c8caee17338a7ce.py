"""Multi-Start L-BFGS-B Optimizer for Rosenbrock."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Global Search via Multi-Start L-BFGS-B with diverse initializations."""

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
        # Use 280s limit to be safe within 300s
        self.deadline = self.start_time + 280.0

        def wrapped_func(x):
            # Check limits
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")

            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            # Update global best
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # Generate diverse starting points using Latin Hypercube Sampling
        rng = np.random.RandomState(42)
        n_starts = min(15, max(5, self.budget // 50))
        
        # LHS for better coverage
        samples = np.empty((n_starts, self.dim))
        for i in range(self.dim):
            perm = rng.permutation(n_starts)
            samples[:, i] = (perm + rng.rand(n_starts)) / n_starts
        samples = samples * 10.0 - 5.0  # Map to [-5, 5]
        
        # Also include origin and some strategic points
        extra_points = [
            np.zeros(self.dim),
            np.ones(self.dim) * 3.0,
            np.ones(self.dim) * -3.0,
        ]
        
        all_starts = [samples] + [np.array([p]) for p in extra_points]
        all_starts = np.vstack(all_starts)
        
        # Sort by distance from origin to prioritize promising starts
        distances = np.linalg.norm(all_starts, axis=1)
        order = np.argsort(distances)
        all_starts = all_starts[order]
        
        # Phase 1: Multi-start L-BFGS-B
        bounds = [(-5.0, 5.0)] * self.dim
        
        for start_idx in range(len(all_starts)):
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            x0 = all_starts[start_idx]
            
            try:
                # Allocate remaining budget for this optimization
                remaining = self.budget - self.calls_made
                maxiter = min(200, max(10, remaining // (self.dim + 1)))
                
                def lbfgsb_callback(xk):
                    return self.calls_made >= self.budget or time.time() > self.deadline
                
                res = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': maxiter,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20
                    },
                    callback=lbfgsb_callback
                )
                
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)
                    
            except StopIteration:
                break
            except Exception:
                continue
        
        # Phase 2: Fine-tune with Nelder-Mead from best point
        if self.best_x is not None and self.calls_made < self.budget:
            try:
                def nm_callback(xk):
                    return self.calls_made >= self.budget or time.time() > self.deadline

                res_nm = scipy.optimize.minimize(
                    wrapped_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    },
                    callback=nm_callback
                )
                
                if res_nm.fun < self.best_f:
                    self.best_f = res_nm.fun
                    self.best_x = np.clip(res_nm.x, -5, 5)
                    
            except StopIteration:
                pass
            except Exception:
                pass

        # If no calls were made (shouldn't happen), return initial guess
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = func(np.clip(self.best_x, -5, 5))

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
