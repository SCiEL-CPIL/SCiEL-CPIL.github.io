"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s limit
        
        # State tracking
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)
        
        bounds = [(-5.0, 5.0)] * self.dim
        
        def wrapped_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            # Ensure x is within bounds, although L-BFGS-B should handle it
            x = np.clip(x, -5.0, 5.0)
            
            val = func(x)
            self.calls += 1
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # Strategy: Multi-start L-BFGS-B with diverse initial points
        # 1. Start at origin (center of box)
        # 2. Sample diverse random points including boundaries
        
        rng = np.random.default_rng()
        
        # Candidate starting points: center + diverse random + boundary points
        starts = [np.zeros(self.dim)]
        
        # Random interior points
        for _ in range(10):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        # Boundary points (corners and edge midpoints)
        for i in range(self.dim):
            for sign in [-1, 1]:
                pt = np.zeros(self.dim)
                pt[i] = sign * 5.0
                starts.append(pt)
        
        # Shuffle to randomize exploration order
        rng.shuffle(starts)
        
        # Track convergence quality to adapt budget allocation
        converged = False
        first_convergence_f = float('inf')
        
        for idx, x0 in enumerate(starts):
            if self.calls >= self.budget or time.time() > deadline:
                break
            
            # Adaptive budget: spend more on first promising start
            remaining = self.budget - self.calls
            if idx == 0:
                maxfun = min(remaining, 200)
                maxiter = 150
            elif not converged and idx < 5:
                maxfun = min(remaining, 100)
                maxiter = 100
            else:
                maxfun = min(remaining, 50)
                maxiter = 50
            
            try:
                # L-BFGS-B is efficient for smooth functions like Sphere
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': maxiter,
                        'maxfun': maxfun,
                        'gtol': 1e-10,
                        'ftol': 1e-12,
                        'maxls': 50
                    }
                )
                
                # Update best from result if available (though wrapper already tracks it)
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
                
                # Check if we've converged to a good solution
                if res.success and not converged:
                    first_convergence_f = self.best_f
                    converged = True
                    
            except (ValueError, TimeoutError, Exception):
                # If an exception is raised due to budget/time, we stop.
                if self.calls >= self.budget or time.time() > deadline:
                    break
                # Otherwise, continue to next start

        # Phase 2: Powell refinement from best point
        if self.calls < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Powell', 
                    bounds=bounds,
                    options={
                        'maxiter': 200, 
                        'maxfev': min(self.budget - self.calls, 150),
                        'ftol': 1e-12,
                        'xtol': 1e-12
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except (ValueError, TimeoutError, Exception):
                pass

        # Phase 3: High-precision L-BFGS-B polish
        if self.calls < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 100, 
                        'maxfun': min(self.budget - self.calls, 100),
                        'gtol': 1e-14,
                        'ftol': 1e-14,
                        'maxls': 100
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except (ValueError, TimeoutError, Exception):
                pass

        # Phase 4: Final Nelder-Mead for any remaining local issues
        if self.calls < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Nelder-Mead', 
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'maxfev': min(self.budget - self.calls, 50),
                        'xatol': 1e-14,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except (ValueError, TimeoutError, Exception):
                pass

        return (self.best_x, self.best_f)
# EVOLVE-BLOCK-END
