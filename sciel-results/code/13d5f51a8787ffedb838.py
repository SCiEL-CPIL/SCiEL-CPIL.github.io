import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        
        # Helper: Evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial diverse sampling to find promising basins
        n_init = 30
        for i in range(n_init):
            if eval_count >= self.budget or time.time() > deadline - 10:
                break
            x = rng.uniform(-5, 5, size=dim)
            evaluate(x)
        
        # Phase 2: Pure (1+1)-ES with 1/5th success rule
        # Start from the best point found in Phase 1
        x = best_x.copy()
        fx = best_f
        
        # Initialize step size based on problem scale
        sigma = 0.5
        success_count = 0
        window_size = max(dim + 1, 10)
        successes_in_window = 0
        window = []
        
        # 1/5th success rule parameters
        c_sigma = 1.0 / (dim + 2)  # Learning rate for sigma
        target_success_rate = 1.0 / 5.0
        
        while eval_count < self.budget - 5 and time.time() < deadline - 10:
            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = evaluate(y)
            
            # Update window for 1/5th rule
            if fy < fx:
                success = True
            else:
                success = False
            
            window.append(success)
            if len(window) > window_size:
                window.pop(0)
            
            if success:
                x = y.copy()
                fx = fy
                successes_in_window += 1
            else:
                # Keep current x
                pass
            
            # Adapt sigma using 1/5th success rule
            if len(window) == window_size:
                success_rate = successes_in_window / window_size
                # Logarithmic update: increase sigma if success rate > 1/5, decrease otherwise
                sigma *= np.exp(c_sigma * (success_rate - target_success_rate))
                
                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-6, 5.0)
                
                # Reset window counter
                successes_in_window = 0
                window = []
            
            # If sigma becomes too small, consider a restart
            if sigma < 1e-5 and eval_count < self.budget - 20:
                # Random restart near best point
                x = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                fx = evaluate(x)
                sigma = 0.3
                window = []
                successes_in_window = 0
        
        return best_x, best_f
