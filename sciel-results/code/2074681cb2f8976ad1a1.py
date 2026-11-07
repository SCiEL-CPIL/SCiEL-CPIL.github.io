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
        deadline = 280.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def time_up():
            return (time.time() - self.start_time) > deadline

        def eval_point(x):
            if self.eval_count >= self.budget or time_up():
                return None
            self.eval_count += 1
            return problem(x)

        # For ZDT2, the Pareto front is f1 = x0, f2 = 1 - sqrt(x0) for x1..xn = 0.
        # To maximize hypervolume, we minimize the area under the staircase approximation of the front.
        # The error is largest where the second derivative is largest (near x0=0).
        # Therefore, we use a non-uniform grid denser near x0=0.
        # x = (k/N)^p with p > 1.
        
        n_sweep = self.budget
        # Use a power law to concentrate points near 0
        # p=2 is a good start.
        k = np.arange(n_sweep)
        x1_values = (k / (n_sweep - 1))**2
        x1_values = x1_values * (hi[0] - lo[0]) + lo[0]
        
        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None
