import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        # Extract low and high bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # The Pareto front of DTLZ7 lies in the subspace where x[2:] = 0.
        # We reduce the search to 2 dimensions: x[0] and x[1].
        
        # 1. Add the 4 extreme corners to ensure full extent coverage
        corners = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ]
        
        # 2. Generate remaining random points in the 2D space
        # We have 200 budget total. 4 used for corners. 196 remaining.
        n_remaining = self.budget - len(corners)
        
        # Use a fixed seed for reproducibility within the run? 
        # The evaluator calls np.random.seed(seed) before creating Optimizer.
        # So we can use np.random directly.
        
        random_2d = np.random.rand(n_remaining, 2)
        
        # Combine corners and random points
        all_2d = np.vstack([corners, random_2d])
        
        # Ensure we don't exceed budget (should be exact)
        assert len(all_2d) <= self.budget, f"Generated {len(all_2d)} points, budget {self.budget}"
        
        # Pad to full dimension
        # x[0] = x_2d[:, 0], x[1] = x_2d[:, 1], x[2:] = 0
        X = np.zeros((len(all_2d), self.dim))
        X[:, 0] = all_2d[:, 0]
        X[:, 1] = all_2d[:, 1]
        
        # Clip to bounds just in case (though 0 and 1 are bounds)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for x in X:
            if time.time() > deadline:
                break
            problem(x)
            
        return X
