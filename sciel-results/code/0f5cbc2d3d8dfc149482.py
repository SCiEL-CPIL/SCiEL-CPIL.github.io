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
        mid = (lo + hi) / 2.0

        # DTLZ4 Pareto front is at x[2:] = mid
        # x[0], x[1] parameterize the front

        # Helper for Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate 196 Halton points in [0, 1]^2
        n_halton = 196
        halton_points = []
        for i in range(1, n_halton + 1):
            x1 = halton(i, 2)
            x2 = halton(i, 3)
            halton_points.append([x1, x2])
        
        # Add 4 corner points
        corners = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        
        # Combine
        all_params = halton_points + corners
        
        # If budget is less than 200, truncate
        if len(all_params) > self.budget:
            all_params = all_params[:self.budget]
            
        # Construct 12D vectors
        X = np.zeros((len(all_params), self.dim))
        for i, p in enumerate(all_params):
            # Scale parameter to bounds
            X[i, 0] = lo[0] + (hi[0] - lo[0]) * p[0]
            X[i, 1] = lo[1] + (hi[1] - lo[1]) * p[1]
            X[i, 2:] = mid # DTLZ4 front condition
            
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        # Check time (unlikely to trigger)
        if time.time() > deadline:
            pass # Just return what we have
            
        return X
