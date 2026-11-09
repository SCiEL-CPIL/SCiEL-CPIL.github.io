import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def eval_func(x):
            nonlocal calls_used
            if calls_used >= self.budget:
                return np.inf
            calls_used += 1
            return func(x)
        
        # Multi-restart strategy with diverse initialization
        # Include origin, random points, and structured points
        restart_points = []
        # Origin
        restart_points.append(np.zeros(self.dim))
        # Random points
        for _ in range(8):
            restart_points.append(rng.uniform(-4, 4, self.dim))
        # Structured points (corners of smaller boxes)
        for _ in range(4):
            restart_points.append(rng.choice([-3, -1, 1, 3], self.dim))
        
        # Adaptive (1+1)-ES for each restart
        for init_x in restart_points:
            if time.time() > deadline:
                break
            if calls_used >= self.budget:
                break
            
            x = init_x.copy()
            fx = eval_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.0
            success_count = 0
            window_size = 10
            successes_in_window = 0
            window_count = 0
            max_steps_for_restart = 50
            
            for step in range(max_steps_for_restart):
                if time.time() > deadline:
                    break
                if calls_used >= self.budget:
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    success_count += 1
                    successes_in_window += 1
                    sigma *= 1.25
                else:
                    sigma *= 0.85
                
                window_count += 1
                if window_count >= window_size:
                    # Adapt sigma based on success rate
                    success_rate = successes_in_window / window_count
                    if success_rate > 0.5:
                        sigma *= 1.5
                    elif success_rate < 0.2:
                        sigma *= 0.5
                    successes_in_window = 0
                    window_count = 0
                
                # Early stopping if sigma is too small
                if sigma < 1e-6:
                    break
        
        # Final polish: do a few more iterations from best_x if budget remains
        if best_x is not None and calls_used < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(min(20, self.budget - calls_used)):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                if fy < fx:
                    x, fx = y, fy
                    best_f = fx
                    best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9
        
        return best_x, best_f
