import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 330s limit
        
        # State tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return best_f
            # Time check
            if time.time() - start_time > time_limit:
                return best_f
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coordinate Descent with Adaptive Step Sizes
        # This exploits the Rosenbrock valley geometry by focusing on 
        # individual coordinates and adapting step sizes based on success.
        
        # Initial point: random within bounds
        x = np.random.uniform(-5, 5, self.dim)
        fx = safe_eval(x)
        
        # Adaptive step sizes for each dimension
        # Start with a moderate step size
        steps = np.ones(self.dim) * 0.5
        
        # Coordinate descent loop
        # We'll cycle through dimensions, trying to improve each one
        # If a step improves, we keep it and potentially increase the step
        # If it doesn't, we shrink the step
        
        while calls_made < self.budget and time.time() - start_time < time_limit:
            improved = False
            for i in range(self.dim):
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break
                    
                # Try positive step
                x_new = x.copy()
                x_new[i] += steps[i]
                x_new = np.clip(x_new, -5, 5)
                
                f_new = safe_eval(x_new)
                
                if f_new < fx:
                    # Improvement
                    x = x_new
                    fx = f_new
                    improved = True
                    # Increase step size (with limit)
                    steps[i] = min(steps[i] * 1.5, 2.0)
                else:
                    # No improvement, try negative step
                    x_new = x.copy()
                    x_new[i] -= steps[i]
                    x_new = np.clip(x_new, -5, 5)
                    
                    f_new = safe_eval(x_new)
                    
                    if f_new < fx:
                        # Improvement
                        x = x_new
                        fx = f_new
                        improved = True
                        # Increase step size (with limit)
                        steps[i] = min(steps[i] * 1.5, 2.0)
                    else:
                        # No improvement in either direction, shrink step
                        steps[i] = max(steps[i] * 0.5, 0.01)
                        
            # If no improvement in a full cycle, we might be stuck
            # Try a small random perturbation to escape local minima
            if not improved:
                # Add small random noise to all dimensions
                noise = np.random.normal(0, 0.1, self.dim)
                x_noisy = np.clip(x + noise, -5, 5)
                f_noisy = safe_eval(x_noisy)
                
                if f_noisy < fx:
                    x = x_noisy
                    fx = f_noisy
                    # Reset some step sizes to help escape
                    steps = np.ones(self.dim) * 0.3
                else:
                    # Shrink all steps further
                    steps = steps * 0.8
                    if np.all(steps < 0.01):
                        # Steps are too small, add larger random jump
                        jump = np.random.uniform(-1, 1, self.dim)
                        x_jump = np.clip(x + jump, -5, 5)
                        f_jump = safe_eval(x_jump)
                        if f_jump < fx:
                            x = x_jump
                            fx = f_jump
                        steps = np.ones(self.dim) * 0.5

        return best_x, best_f
