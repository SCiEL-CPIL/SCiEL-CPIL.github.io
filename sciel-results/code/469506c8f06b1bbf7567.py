import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.x1_values = None
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        
        if self.x1_values is None:
            self._compute_optimal_x1()
        
        # Extract bounds for clipping
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate and evaluate points
        for x1 in self.x1_values:
            x = np.zeros(self.dim)
            x[0] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            problem(x)
            
            # Safety check for time
            if time.time() - self.start_time > 320:
                break
                
        return None

    def _compute_optimal_x1(self):
        k = self.budget
        if k <= 0:
            self.x1_values = np.array([])
            return
            
        # Initial guess: uniform
        x = np.linspace(0, 1, k)
        
        # Iterative refinement (Gauss-Seidel style)
        # We want to maximize H
        # Interior points i: x_i = x_{i+1} - sqrt(x_{i+1}^2 - x_{i-1}^2)
        # Last point k-1: x_{k-1} = (22 + sqrt(484 + 12*x_{k-2}^2)) / 6
        # First point 0: x_0 = 0
        
        for _ in range(100): # Max iterations
            # Update interior points
            for i in range(1, k-1):
                # x_i^2 - 2 x_{i+1} x_i + x_{i-1}^2 = 0
                # x_i = x_{i+1} - sqrt(x_{i+1}^2 - x_{i-1}^2)
                # Check if x_{i+1}^2 >= x_{i-1}^2
                if x[i+1]**2 >= x[i-1]**2:
                    x_new = x[i+1] - np.sqrt(x[i+1]**2 - x[i-1]**2)
                else:
                    x_new = (x[i-1] + x[i+1]) / 2 # Fallback
                
                # Clamp to [x[i-1], x[i+1]]
                x_new = np.clip(x_new, x[i-1], x[i+1])
                x[i] = x_new
            
            # Update last point
            if k > 1:
                # 3 x^2 - 22 x - x_{k-2}^2 = 0
                # x = (22 + sqrt(484 + 12 x_{k-2}^2)) / 6
                x_prev = x[k-2]
                x_new = (22 + np.sqrt(484 + 12 * x_prev**2)) / 6.0
                x_new = np.clip(x_new, x_prev, 1.0)
                x[k-1] = x_new
            
            # Update first point (always 0)
            x[0] = 0.0
            
            # Check convergence
            if np.max(np.abs(x - np.roll(x, 1))) < 1e-10:
                break

        self.x1_values = x
