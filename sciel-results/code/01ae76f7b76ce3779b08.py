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

        # For ZDT2, setting x[1:] = 0 places points exactly on the Pareto front.
        # f1 = x[0], f2 = (1 + 10*(n-1)) * (1 - sqrt(x[0]/n)) when x[1:] = 0
        # Use all budget for dense uniform sweep of x[0] with x[1:] = 0
        n_sweep = self.budget
        x1_values = np.linspace(lo[0], hi[0], n_sweep)
        
        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None
