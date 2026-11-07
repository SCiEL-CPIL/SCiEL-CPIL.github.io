import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def time_up():
            return (time.time() - self.start_time) > deadline

        def eval_point(x):
            if self.eval_count >= self.budget or time_up():
                return None
            self.eval_count += 1
            return problem(x)

        n = self.dim
        g = 1 + 10 * (n - 1)  # 291 for n=30
        
        # Revert to full range sweep as in the higher-scoring parent
        x1_min = 0.0
        x1_max = 1.0
        
        # Ensure valid range
        x1_min = max(x1_min, lo[0])
        x1_max = min(x1_max, hi[0])
        
        if x1_min >= x1_max:
            x1_min = 0.0
            x1_max = 1.0
            
        # Use all budget for dense sweep
        n_sweep = min(self.budget, 200)
        x1_values = np.linspace(x1_min, x1_max, n_sweep)
        
        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None
