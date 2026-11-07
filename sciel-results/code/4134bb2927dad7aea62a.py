"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with time and budget guards."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.budget_limit = budget
        self.start_time = None
        self.best_x = None
        self.best_f = float('inf')
        self.calls = 0
        # Safety margin: Evaluator kills at 330s. We stop at 290s to be safe.
        self.time_limit = 290.0

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        dim = self.dim
        bounds = [(-5.0, 5.0)] * dim
        rng = np.random.default_rng(42)

        # Wrapped function to track calls, time, and best
        def wrapped_func(x):
            self.calls += 1
            if self.calls > self.budget_limit:
                raise Exception("Budget exceeded")
            
            current_time = time.time()
            if current_time - self.start_time > self.time_limit:
                raise Exception("Time limit exceeded")

            # Clip to bounds
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
                
            return fx

        # Generate 3 initial points
        initial_points = []
        initial_values = []
        
        for _ in range(3):
            if self.calls >= self.budget_limit:
                break
            x0 = rng.uniform(-5.0, 5.0, dim)
            try:
                f0 = wrapped_func(x0)
                initial_points.append(x0)
                initial_values.append(f0)
            except Exception:
                break

        if not initial_points:
            # Fallback if no evaluations were possible
            return np.zeros(dim), float('inf')

        # Sort by initial value
        sorted_indices = np.argsort(initial_values)
        
        # Run L-BFGS-B from the best initial points
        remaining = self.budget - self.calls
        
        if remaining <= 0:
            return self.best_x, self.best_f

        # Allocate budget: 70% to the best start, 30% to the second best
        alloc1 = int(remaining * 0.7)
        alloc2 = remaining - alloc1
        
        # Estimate maxiter: each iteration costs ~ dim+1 function calls for gradient
        maxiter1 = max(1, alloc1 // (dim + 1))
        
        try:
            minimize(wrapped_func, initial_points[sorted_indices[0]], 
                     method='L-BFGS-B', bounds=bounds, 
                     options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-10})
        except Exception:
            pass

        remaining = self.budget - self.calls
        if remaining > 0 and len(sorted_indices) > 1:
            maxiter2 = max(1, remaining // (dim + 1))
            try:
                minimize(wrapped_func, initial_points[sorted_indices[1]], 
                         method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': maxiter2, 'ftol': 1e-12, 'gtol': 1e-10})
            except Exception:
                pass

        # If we still have budget and time, and a 3rd point, use it
        remaining = self.budget - self.calls
        if remaining > 0 and len(sorted_indices) > 2:
            maxiter3 = max(1, remaining // (dim + 1))
            try:
                minimize(wrapped_func, initial_points[sorted_indices[2]], 
                         method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': maxiter3, 'ftol': 1e-12, 'gtol': 1e-10})
            except Exception:
                pass

        if self.best_x is None:
             self.best_x = initial_points[sorted_indices[0]]
             self.best_f = initial_values[sorted_indices[0]]

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
