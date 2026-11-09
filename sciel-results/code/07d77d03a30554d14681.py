import numpy as np
import time
import scipy.optimize
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin for 300s limit
        
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.bounds = [(-5.0, 5.0)] * self.dim

        def wrapped_func(x):
            # Check time
            if time.time() > deadline:
                return 1e100 # Return a very large value to stop optimization
            
            # Check budget
            if self.call_count >= self.budget:
                return 1e100
            
            self.call_count += 1
            
            # Ensure x is a numpy array of correct shape
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            
            # Clip to box just in case, although func does it too
            x = np.clip(x, -5.0, 5.0)
            
            try:
                fx = float(func(x))
            except Exception:
                fx = 1e100
            
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
                
            return fx

        # Phase 1: Multi-Start Nelder-Mead
        # Use a Sobol sequence for diverse initial points
        # SciPy has qmc for Sobol
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            # Generate 20 points
            n_starts = 20
            points = sampler.random(n_starts) * 10.0 - 5.0 # Map [0,1] to [-5,5]
        except ImportError:
            # Fallback to random if qmc not available (though it is in recent scipy)
            rng = np.random.default_rng(0)
            points = rng.uniform(-5.0, 5.0, size=(20, self.dim))

        # Allocate budget per start
        # 20 starts * 15 calls = 300 calls
        budget_per_start = 15
        
        for p in points:
            if self.call_count >= self.budget - 50: # Keep 50 for final polish
                break
            if time.time() > deadline:
                break
                
            # Run Nelder-Mead from this point
            try:
                res = minimize(
                    wrapped_func, 
                    p, 
                    method='Nelder-Mead',
                    options={'maxiter': budget_per_start, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                # The wrapped_func updates self.best_x and self.best_f automatically
            except Exception:
                pass

        # Phase 2: Local Polish with Nelder-Mead (or Powell)
        # Remaining budget
        remaining_budget = self.budget - self.call_count
        
        if remaining_budget > 10 and time.time() < deadline:
            # Start from the best found so far
            if self.best_x is None:
                x0 = np.zeros(self.dim)
            else:
                x0 = self.best_x.copy()
                
            try:
                # Use Powell's method for final polish? It can be more robust than NM for final steps.
                # Or just NM with tighter tolerances.
                res_nm = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': remaining_budget, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if res_nm.fun < self.best_f:
                    self.best_x = res_nm.x
                    self.best_f = res_nm.fun

            except Exception:
                pass

        # Final return
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = wrapped_func(self.best_x)
            
        return self.best_x, self.best_f
