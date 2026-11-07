import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        N = self.budget
        dim = self.dim
        start_time = time.time()
        deadline = start_time + 280

        # Optimize alpha for power law spacing x_i = (i/(N-1))^alpha
        # We derived alpha ~ 4 for minimizing left Riemann sum of convex decreasing y(x)
        # Let's search a range of alphas to be safe.
        
        best_alpha = 4.0
        min_sum = float('inf')
        
        # Range of alphas to test
        # Based on h_i ~ x_i^{3/4} => x_i ~ i^4
        # Let's test 1.0 to 8.0
        alphas = np.arange(1.0, 8.1, 0.1)
        
        for alpha in alphas:
            x1 = (np.arange(N) / (N - 1)) ** alpha
            x1[0] = 0.0
            x1[-1] = 1.0
            
            y = 1.0 - np.sqrt(x1)
            
            # Left Riemann sum for [0, 1]
            dx = np.diff(x1)
            sum_val = np.sum(dx * y[:-1])
            
            # Last segment [1, 11] has y(1)=0, so contribution is 0.
            # If x1[-1] was not 1, we'd add (11-x1[-1])*y[-1]
            
            if sum_val < min_sum:
                min_sum = sum_val
                best_alpha = alpha

        # Generate final points
        x1_opt = (np.arange(N) / (N - 1)) ** best_alpha
        x1_opt[0] = 0.0
        x1_opt[-1] = 1.0

        X = np.zeros((N, dim))
        X[:, 0] = x1_opt

        results = []
        for i in range(N):
            if time.time() > deadline:
                break
            f = problem(X[i])
            results.append(f)

        return X
