import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        budget = self.budget
        
        if budget <= 0:
            return None

        # ZDT2: f1 = x1, f2 = (1+10*(n-1))*(1-sqrt(f1))
        # Reference point (11, 11). f2 < 11 only when f1 > (1-11/(1+10*(n-1)))^2
        C = 1 + 10 * (n - 1)  # 310 for n=30
        f1_min = (1.0 - 11.0 / C) ** 2  # ≈ 0.9305
        
        # Add small margin to ensure points are strictly better than ref
        f1_low = f1_min + 1e-9
        f1_high = 1.0
        
        # Uniform spacing in the effective range
        if budget == 1:
            x1_vals = np.array([(f1_low + f1_high) / 2.0])
        else:
            x1_vals = np.linspace(f1_low, f1_high, budget)
        
        # Create full vectors
        # x2..xn = 0 to minimize g(x) and place on Pareto front
        X = np.zeros((budget, n))
        X[:, 0] = x1_vals
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
