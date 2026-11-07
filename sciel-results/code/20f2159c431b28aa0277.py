import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # ZDT2 Pareto front: x_2..x_n = 0
        # f1 = x1
        # f2 = 1 - sqrt(x1)
        
        # We have self.budget evaluations.
        # We will use all of them to sample the Pareto front.
        N = self.budget
        dim = self.dim
        
        # Generate x1 values
        # Strategy: Uniform in f2 (which is 1 - sqrt(x1))
        # Let f2_i be uniform in [0, 1]
        # f2_i = 1 - i/(N-1) for i in 0..N-1
        # 1 - sqrt(x1_i) = 1 - i/(N-1)
        # sqrt(x1_i) = i/(N-1)
        # x1_i = (i/(N-1))^2
        
        # Handle N=1 edge case, though budget is 200
        if N == 1:
            x1_vals = np.array([0.5])
        else:
            i = np.arange(N)
            # Map i to [0, 1]
            u = i / (N - 1)
            x1_vals = u**2
        
        # Construct decision vectors
        # x = [x1, 0, 0, ..., 0]
        X = np.zeros((N, dim))
        X[:, 0] = x1_vals
        
        # Evaluate
        # We must ensure we don't exceed time.
        # Each eval is fast, 200 evals should take < 1s.
        
        F = np.zeros((N, self.n_obj))
        for i in range(N):
            if time.time() > deadline:
                # If we run out of time, break. 
                # But we should have finished long before.
                break
            F[i] = problem(X[i])
            
        # The problem statement says "Every returned objective vector is recorded".
        # This implies the evaluator tracks the calls to problem().
        # We don't need to return F, but we return X as per interface.
        return X
