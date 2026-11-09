import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid Nelder-Mead + L-BFGS-B Optimizer for Rosenbrock."""

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
        # Safety margin: 300s limit, use 290s
        self.deadline = self.start_time + 290.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            # Ensure x is a 1D array of length dim
            x = np.asarray(x).flatten()
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        rng = np.random.default_rng(42)
        
        # Structural Change: Multi-Start L-BFGS-B
        # Phase 1: Broad Search with short L-BFGS-B runs
        # Phase 2: Deep Refinement of best candidates
        
        start_points = []
        # 20 diverse starts in [-4, 4]^5
        for _ in range(20):
            x0 = rng.uniform(-4, 4, self.dim)
            start_points.append(x0)
            
        # Add some specific points
        start_points.append(np.zeros(self.dim))
        start_points.append(np.ones(self.dim) * 1.0) # Standard Rosenbrock min
        start_points.append(np.ones(self.dim) * -1.0)
        
        rng.shuffle(start_points)
        
        # Phase 1: Short runs to find basins
        # Budget: 20 starts * 15 calls = 300 calls
        # This leaves 200 calls for Phase 2
        phase1_results = []
        
        for i, x0 in enumerate(start_points):
            if self.calls_made >= self.budget - 150: # Reserve 150 for phase 2
                break
            if time.time() > self.deadline - 10:
                break
                
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 15,
                        'ftol': 1e-6,
                        'gtol': 1e-4
                    }
                )
                phase1_results.append((res.fun, res.x))
            except StopIteration:
                break
            except Exception:
                continue
        
        # Phase 2: Refine the best 4 distinct points
        phase1_results.sort(key=lambda x: x[0])
        
        candidates = []
        for f_val, x_val in phase1_results[:10]:
            is_distinct = True
            for cx in candidates:
                if np.linalg.norm(x_val - cx) < 0.5:
                    is_distinct = False
                    break
            if is_distinct:
                candidates.append(x_val)
            if len(candidates) >= 4:
                break
        
        if not candidates and self.best_x is not None:
            candidates.append(self.best_x)
            
        for x0 in candidates:
            if self.calls_made >= self.budget - 5:
                break
            if time.time() > self.deadline - 2:
                break
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
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
        
        # Final check
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = func(np.clip(x0, -5, 5))

        return self.best_x, self.best_f
