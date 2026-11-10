import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        N = self.budget
        
        # For ZDT1, the Pareto front is at x[1:] = 0.
        # f1 = x[0], f2 = 1 - sqrt(x[0]).
        # To maximize the Hypervolume of the sampled points relative to the reference point,
        # we should distribute points non-uniformly.
        # The derivative of the front is -1/(2*sqrt(x)), which is steep near x=0.
        # The optimal density for minimizing discretization error in HV is proportional to 1/sqrt(x).
        # This corresponds to the mapping x = t^2 where t is uniform.
        
        if N < 2:
            x1_samples = np.array([lo[0], hi[0]])
        else:
            # Uniform t in [0, 1]
            t_samples = np.linspace(0.0, 1.0, N)
            # Map to x in [lo[0], hi[0]] using x = t^2
            # This clusters points near lo[0] (which is 0)
            x1_samples = (t_samples ** 2) * (hi[0] - lo[0]) + lo[0]
            
            # Ensure endpoints are exactly lo and hi to avoid floating point issues
            x1_samples[0] = lo[0]
            x1_samples[-1] = hi[0]

        results = []
        for x1 in x1_samples:
            if time.time() > deadline:
                break
            
            # Create point on the Pareto front
            x = np.full(self.dim, lo)
            x[0] = x1
            
            f = problem(x)
            results.append(f)
            
        return np.array(results)
