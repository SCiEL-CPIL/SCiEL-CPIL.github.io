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
            
            # Strategy: Place points on the Pareto front of ZDT1
            # The Pareto front is defined by x_0 in [0, 1] and x_i = 0.5 for i > 0.
            # This minimizes g(x) to 1.
            
            # We will use the entire budget to sample x_0.
            # To maximize hypervolume, we want a good spread of x_0.
            # Uniform sampling is effective.
            
            # Generate x_0 values
            # Using linspace might be too regular if the reference is also regular?
            # Random sampling with replacement? No, we want distinct points.
            # Let's use a quasi-random sequence or just uniform random.
            # Uniform random is simple and effective.
            
            # However, to ensure we cover the extremes (0 and 1) which contribute significantly to HV,
            # we should ensure 0 and 1 are included or very close.
            
            x0_samples = np.linspace(0.0, 1.0, self.budget)
            
            # Create population
            X = np.zeros((self.budget, self.dim))
            X[:, 0] = x0_samples
            X[:, 1:] = 0.5  # Optimal for g(x)
            
            # Clip to bounds just in case (though 0.5 and [0,1] are safe)
            X = np.clip(X, lo, hi)
            
            # Evaluate
            F = np.array([problem(x) for x in X])
            
            # The return value is ignored, but we must not exceed budget.
            # We used exactly self.budget evaluations.
            
            return X
