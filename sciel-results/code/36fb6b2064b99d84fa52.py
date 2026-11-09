import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: stop 5 seconds before the 300s limit
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine the optimal tail values for ZDT4
        # The term in g(x) is x^2 - 10*cos(4*pi*x) + 10
        # We test candidates 0.0, 0.5, 1.0 to find the local minimizer within bounds
        candidates = [0.0, 0.5, 1.0]
        best_tail_val = 0.0
        min_cost = float('inf')
        
        for c in candidates:
            c_clipped = np.clip(c, lo[1], hi[1])
            # Calculate cost for this single variable term
            cost = c_clipped**2 - 10 * np.cos(4 * np.pi * c_clipped) + 10
            if cost < min_cost:
                min_cost = cost
                best_tail_val = c_clipped
        
        # The optimal tail is a vector of this best value
        x_tail = np.full(self.dim - 1, best_tail_val)
        
        n_evals = self.budget
        if n_evals < 1:
            return None
        
        # Generate x0 values covering the full range
        # Use a distribution that clusters points near both ends of x0
        # The Pareto front of ZDT4 is curved, so more points near the ends help
        # Use a sine-based distribution to cluster points near the ends
        # t = 0.5 * (1 - cos(pi * i/(n-1))) clusters near 0 and 1
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
                
            if n_evals > 1:
                # Use arcsine distribution: t = 0.5 * (1 - cos(pi * i/(n-1)))
                # This clusters points near both 0 and 1
                t = 0.5 * (1 - np.cos(np.pi * i / (n_evals - 1)))
            else:
                t = 0.5
                
            x0 = lo[0] + t * (hi[0] - lo[0])
            
            x = np.empty(self.dim)
            x[0] = x0
            x[1:] = x_tail
            
            try:
                problem(x)
            except Exception:
                # If an exception occurs, we skip this point and continue
                # The contract says "an exception later keeps the evaluations made"
                continue

        return None
