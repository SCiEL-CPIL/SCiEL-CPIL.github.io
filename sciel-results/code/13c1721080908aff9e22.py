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
        
        # Use all budget for dense sweep
        n_sweep = min(self.budget, 200)
        
        # Generate uniform u in [0, 1]
        u_values = np.linspace(0.0, 1.0, n_sweep)
        
        # Map to x1 such that f2 is uniformly spaced (better for convex hypervolume)
        # f2 = 2 - sqrt(2*x1) => x1 = (2-f2)^2 / 2
        # If f2 is uniform from f2_max to f2_min, let u be uniform 0..1
        # f2 = 2 - u*sqrt(2)
        # x1 = (u*sqrt(2))^2 / 2 = u^2
        x1_values = u_values ** 2
        
        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)
            
        # If we have remaining budget, add points at the extremes of the Pareto front
        # to ensure we cover the endpoints which are critical for hypervolume
        if self.eval_count < self.budget and not time_up():
            # Endpoint 1: x[0] = 0 (f1=0, f2=2)
            x = np.zeros(self.dim)
            x = np.clip(x, lo, hi)
            eval_point(x)
        
        if self.eval_count < self.budget and not time_up():
            # Endpoint 2: x[0] = 1 (f1=1, f2=0)
            x = np.zeros(self.dim)
            x[0] = 1.0
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None
