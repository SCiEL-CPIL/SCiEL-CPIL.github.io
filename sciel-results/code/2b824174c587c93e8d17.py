import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290

        best_x = None
        best_f = float('inf')
        eval_count = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Random Initialization (10% of budget)
        n_init = int(0.1 * budget)
        rng = np.random.default_rng(42)
        
        for _ in range(n_init):
            if eval_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            f = func(x)
            eval_count += 1
            update_best(x, f)

        if eval_count >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        success_rate = 0.0
        # Adaptive step size based on success rate
        # Target success rate ~ 0.33 (1/3 rule) or similar
        # Simple adaptation: if success > 0.3, increase sigma, else decrease
        
        while eval_count < budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            eval_count += 1
            
            if fy < fx:
                x, fx = y, fy
                update_best(x, fx)
                sigma *= 1.2
            else:
                sigma *= 0.9
            
            # Prevent sigma from exploding or vanishing
            if sigma > 10.0:
                sigma = 10.0
            if sigma < 1e-6:
                sigma = 1e-6

        return best_x, best_f
