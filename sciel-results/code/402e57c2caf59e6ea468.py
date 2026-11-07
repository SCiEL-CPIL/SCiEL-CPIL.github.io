"""Rewritten optimizer for BBOB-style f14 (Different powers) using L-BFGS-B."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer with strict budget and time guards."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = time.time()
        self.time_limit = 280.0  # Safety margin under 330s
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.best_x = None
        self.best_f = float('inf')
        
        dim = self.dim
        budget = self.budget
        
        def wrapped_func(x):
            self.call_count += 1
            # Time guard
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError("Time limit exceeded")
            # Budget guard
            if self.call_count > budget:
                raise ValueError("Budget exceeded")
            
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
                
            return fx

        # 1. Generate deterministic Halton sequence for initial points
        # We use 10 initial points to cover the space well.
        n_initial = 10
        halton_points = self._halton_sequence(n_initial, dim)
        
        # Evaluate initial points
        initial_results = []
        for p in halton_points:
            try:
                fx = wrapped_func(p)
                initial_results.append((fx, p))
            except (ValueError, TimeoutError):
                break
        
        if self.best_x is None:
            # Fallback if something went wrong immediately
            return np.zeros(dim), float('inf')

        # Sort initial results by fitness
        initial_results.sort(key=lambda x: x[0])
        
        # 2. Run L-BFGS-B from the best initial points
        # We try to run L-BFGS-B from the top 3 initial points if budget allows.
        # L-BFGS-B typically converges in < 100 evaluations for 5D smooth functions.
        
        bounds = [(-5.0, 5.0)] * dim
        
        # Reserve some budget for safety
        reserve = 20
        
        for i, (fx, x0) in enumerate(initial_results):
            # Check if we have enough budget for a reasonable L-BFGS-B run
            # A typical L-BFGS-B run might take 50-100 evals.
            if self.call_count + 50 > budget - reserve:
                break
                
            if time.time() - self.start_time > self.time_limit - 10:
                break

            try:
                # Run L-BFGS-B
                # maxiter=50 is usually sufficient for 5D BBOB functions
                # tol=1e-10 for high precision
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # The result is already tracked in self.best_f/x via wrapped_func
                # If the optimization didn't improve, we just move to the next start point
                
            except (ValueError, TimeoutError, Exception):
                # If budget or time was exceeded during optimization, 
                # we break out of the loop. The best found so far is in self.best_x
                break

        return self.best_x, self.best_f

    def _halton_sequence(self, n, dim):
        """Generate Halton sequence points in [0, 1)^dim, scaled to [-5, 5]^dim."""
        points = []
        for i in range(1, n + 1):
            point = []
            for j in range(dim):
                base = j + 2  # Start with base 2, 3, 5, ...
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    f /= base
                    r += f * (i_val % base)
                    i_val //= base
                point.append(r)
            # Scale from [0, 1) to [-5, 5]
            point = np.array(point) * 10.0 - 5.0
            points.append(point)
        return points
# EVOLVE-BLOCK-END
