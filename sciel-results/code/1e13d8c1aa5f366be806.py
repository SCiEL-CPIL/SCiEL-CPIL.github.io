import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in scoring? 
        # The task says "hidden seeded shift...". The optimizer itself should probably be deterministic or use a fixed seed to ensure consistent behavior if re-run?
        # The prompt says "For each instance... evaluator creates Optimizer...". 
        # Usually, we want the optimizer to be deterministic given the same func.
        # Using a fixed seed ensures that if the func is the same, we do the same thing.
        
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 320.0 # 330s kill, keep 10s margin
        
        # Phase 1: Initial Sampling (Cheap Global Exploration)
        # Try a few random points to find a good starting point
        n_init = 5
        best_x = None
        best_f = np.inf
        
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        calls = n_init
        
        # Phase 2: (1+1)-ES with Restarts
        x = best_x.copy()
        fx = best_f
        sigma = 2.0 # Initial step size
        
        success_window = [] # Track successes in last 15 steps
        last_improvement_call = calls
        stagnant_count = 0
        
        while calls < budget:
            # Time Guard
            if time.time() - start_time > time_limit:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            
            # Update best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                last_improvement_call = calls
                stagnant_count = 0
            
            # (1+1)-ES Logic
            if fy < fx:
                x = y
                fx = fy
                success_window.append(1)
            else:
                success_window.append(0)
                
            # Keep window size 15
            if len(success_window) > 15:
                success_window.pop(0)
                
            # Adapt sigma
            if len(success_window) == 15:
                success_rate = np.mean(success_window)
                if success_rate > 0.35:
                    sigma *= 1.3
                elif success_rate < 0.15:
                    sigma *= 0.7
            
            # Stagnation / Restart Logic
            # If no improvement in last 30 steps, try a leap
            if calls - last_improvement_call >= 30:
                stagnant_count += 1
                if stagnant_count >= 3:
                    # Escape local minimum with a leap
                    leap_size = max(sigma * 3, 2.0)
                    x = x + rng.normal(0, leap_size, dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    sigma = max(sigma * 0.5, 0.5)
                    success_window = []
                    stagnant_count = 0
                    last_improvement_call = calls
                    # Update best if this leap was good
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Try smaller steps to escape
                    sigma = max(sigma * 0.7, 0.1)
        
        # Phase 3: Final Polish - refine the best solution found
        if calls < budget and best_x is not None:
            polish_x = best_x.copy()
            polish_f = best_f
            polish_sigma = 0.1
            
            # Use remaining budget for fine-tuning
            remaining = budget - calls
            polish_steps = min(remaining, 50)
            
            for _ in range(polish_steps):
                if time.time() - start_time > time_limit:
                    break
                if calls >= budget:
                    break
                
                y = polish_x + polish_sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < polish_f:
                    polish_x = y
                    polish_f = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    polish_sigma *= 1.1
                else:
                    polish_sigma *= 0.9
                
                # Keep sigma in reasonable bounds
                polish_sigma = max(0.01, min(polish_sigma, 1.0))
        
        return best_x, best_f
