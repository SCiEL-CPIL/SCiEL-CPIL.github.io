import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = float(fx)
                best_x = x.copy()
            return float(fx)

        # Phase 1: Quick random sampling to find a good starting region
        n_random = min(20, budget // 10)
        for _ in range(n_random):
            if eval_count >= budget - 20 or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, d)
            fx = evaluate(x)

        # Phase 2: Powell's method - coordinate-wise line search
        # Initialize directions with coordinate axes
        directions = np.eye(d)
        x_current = best_x.copy()
        f_current = best_f
        
        # Powell's method iterations
        max_powell_iter = 30
        for _iter in range(max_powell_iter):
            if eval_count >= budget - 10 or time.time() > deadline:
                break
            
            x_temp = x_current.copy()
            f_temp = f_current
            
            # Line search along each direction
            for i in range(d):
                if eval_count >= budget - 10 or time.time() > deadline:
                    break
                
                # Perform 1D line search along direction i
                dir_i = directions[i]
                
                # Find a good step size by trying several steps
                # Use a golden-section-like approach with limited evaluations
                best_step = 0.0
                best_f_dir = f_temp
                
                # Try positive and negative directions with adaptive step sizes
                for step_mag in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]:
                    if eval_count >= budget - 10 or time.time() > deadline:
                        break
                    
                    # Try +step
                    x_try = x_temp + step_mag * dir_i
                    x_try = np.clip(x_try, -5, 5)
                    f_try = evaluate(x_try)
                    if f_try < best_f_dir:
                        best_f_dir = f_try
                        best_step = step_mag
                    
                    # Try -step
                    x_try = x_temp - step_mag * dir_i
                    x_try = np.clip(x_try, -5, 5)
                    f_try = evaluate(x_try)
                    if f_try < best_f_dir:
                        best_f_dir = f_try
                        best_step = -step_mag
                
                # Apply best step
                if best_step != 0.0:
                    x_temp = x_temp + best_step * dir_i
                    f_temp = best_f_dir
            
            # Check if we made progress
            if f_temp < f_current - 1e-10:
                # Update the last direction to be the net displacement
                directions[d-1] = x_temp - x_current
                x_current = x_temp
                f_current = f_temp
                
                # Reorder directions: move the one with largest reduction to the end
                # For simplicity, just keep the net displacement as last direction
            else:
                # No improvement, break
                break
            
            # Update best
            if f_current < best_f:
                best_f = f_current
                best_x = x_current.copy()

        # Phase 3: Final coordinate-wise refinement with smaller steps
        refine_x = best_x.copy()
        refine_f = best_f
        for iteration in range(5):
            if eval_count >= budget - 5 or time.time() > deadline:
                break
            improved = False
            for i in range(d):
                if eval_count >= budget - 5 or time.time() > deadline:
                    break
                for step in [0.05, 0.02, 0.01, 0.005, 0.001, -0.05, -0.02, -0.01, -0.005, -0.001]:
                    if eval_count >= budget - 5 or time.time() > deadline:
                        break
                    test_x = refine_x.copy()
                    test_x[i] += step
                    test_x = np.clip(test_x, -5, 5)
                    fx = evaluate(test_x)
                    if fx < refine_f:
                        refine_f = fx
                        refine_x = test_x.copy()
                        best_f = fx
                        best_x = refine_x.copy()
                        improved = True
                        break
            if not improved:
                break
        
        return best_x, best_f
