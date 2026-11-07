"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """L-BFGS-B optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
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

        # Strategy: Coordinate-wise derivative-free line search (Powell-like)
        # 1. Start at origin
        # 2. Use cyclic coordinate descent with adaptive step sizes
        # 3. This avoids geometric coupling of Nelder-Mead and allows rapid convergence
        #    along hidden rotated axes of the Sphere function
        
        rng = np.random.default_rng(42)
        
        # Start at origin
        x = np.zeros(self.dim)
        val = wrapped_func(x)
        
        # Initialize step sizes
        step_sizes = np.ones(self.dim) * 0.5
        
        # Powell-like cyclic coordinate descent
        # For each dimension, do a 1D line search
        max_rounds = 50
        for round_num in range(max_rounds):
            if self.calls >= self.budget or time.time() > deadline:
                break
            
            for i in range(self.dim):
                if self.calls >= self.budget or time.time() > deadline:
                    break
                
                # 1D line search along dimension i
                best_xi = x[i]
                best_val_i = val
                
                # Try a few step sizes in both directions
                for step_mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, -0.01, -0.05, -0.1, -0.25, -0.5, -1.0]:
                    trial_x = x.copy()
                    trial_x[i] = x[i] + step_sizes[i] * step_mult
                    trial_x = np.clip(trial_x, -5.0, 5.0)
                    
                    trial_val = wrapped_func(trial_x)
                    
                    if trial_val < best_val_i:
                        best_val_i = trial_val
                        best_xi = trial_x[i]
                
                # Update x with best found in this dimension
                if best_xi != x[i]:
                    x[i] = best_xi
                    val = best_val_i
                    
                    if val < self.best_f:
                        self.best_f = val
                        self.best_x = x.copy()
                
                # Adapt step size based on whether we moved
                if best_xi != x[i] - step_sizes[i] * 0:  # We moved
                    # Increase step size slightly if we made progress
                    step_sizes[i] *= 1.2
                else:
                    # Decrease step size if no progress
                    step_sizes[i] *= 0.7
                
                # Keep step sizes within reasonable bounds
                step_sizes[i] = np.clip(step_sizes[i], 0.001, 5.0)
        
        # Final high-precision polish pass using Nelder-Mead
        if self.calls < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Nelder-Mead', 
                    bounds=bounds,
                    options={
                        'maxiter': 500, 
                        'maxfev': self.budget - self.calls,
                        'xatol': 1e-10,
                        'fatol': 1e-10
                    }
                )
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except (ValueError, TimeoutError, Exception):
                pass

        return (self.best_x, self.best_f)
# EVOLVE-BLOCK-END
