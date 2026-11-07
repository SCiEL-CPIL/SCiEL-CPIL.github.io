import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300 * 0.9  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG4 specific structure:
        # f1 = x1
        # f2 = x2
        # f3 = 1/(x1*x2) * g(x3..xn)
        # Pareto front is when g(x) is minimized (usually 1).
        # We sample x1, x2 to cover the front, and try different values for x3..xn
        # to find the minimum g.
        
        # Valid range for x1, x2 to keep f3 < 7 (ref point)
        # 1/(x1*x2) < 7 => x1*x2 > 1/7
        # We sample log-uniformly in [0.1, 1.0] to ensure product is often > 0.14
        # and to cover the shape of the hyperbola well.
        
        n_evals = 0
        all_points = []
        all_f = []
        
        # We will split the budget into 3 batches for x3..xn: lo, mid, hi
        # This hedges against not knowing the exact minimizer of g(x).
        batch_size = self.budget // 3
        if batch_size < 10: batch_size = 10
        
        candidates = [
            np.zeros(self.dim - 2),  # lo (assuming lo is 0, or we use actual lo)
            np.ones(self.dim - 2) * 0.5, # mid
            np.ones(self.dim - 2)       # hi
        ]
        
        # Adjust candidates to actual bounds
        # If bounds are not [0,1], we should scale or just use lo/hi/mid of bounds
        lo_rest = lo[2:]
        hi_rest = hi[2:]
        mid_rest = (lo_rest + hi_rest) / 2
        
        candidates = [
            lo_rest,
            mid_rest,
            hi_rest
        ]
        
        for c in candidates:
            for i in range(batch_size):
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Sample x1, x2 log-uniformly
                # Use np.random.uniform for log space
                log_lo = np.log(max(0.01, lo[0]))
                log_hi = np.log(hi[0])
                x1 = np.exp(np.random.uniform(log_lo, log_hi))
                
                log_lo2 = np.log(max(0.01, lo[1]))
                log_hi2 = np.log(hi[1])
                x2 = np.exp(np.random.uniform(log_lo2, log_hi2))
                
                # Clip to bounds
                x1 = np.clip(x1, lo[0], hi[0])
                x2 = np.clip(x2, lo[1], hi[1])
                
                x = np.concatenate(([x1, x2], c))
                
                f = problem(x)
                all_points.append(x)
                all_f.append(f)
                n_evals += 1
        
        # If we have budget left, we can do a simple local search or just return
        # Given the strong prior, random local search is less effective than just
        # having more samples on the front.
        
        # Convert to arrays for return (though return value is ignored, 
        # the task says "Every returned objective vector is recorded", 
        # wait, "The return value of __call__ is ignored." 
        # "Every returned objective vector is recorded" - this refers to the 
        # output of `problem(x)` calls? 
        # "problem(x) ... returns 3 objective values."
        # "Every returned objective vector is recorded" - likely means the evaluator
        # intercepts the calls to `problem` and records the outputs.
        # So I just need to make sure I call `problem` and don't crash.
        
        return all_points
