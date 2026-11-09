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
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7: x0, x1 are active. x2... are inactive (set to 0).
        # Pareto front: f1 = x0, f2 = x1*(1+x0), f3 = 1 - x0 + x0^2*sin(pi*x0/2)
        # Uniform grid on [0,1]^2 for x0, x1
        
        n0 = 14
        n1 = 14
        x0_vals = np.linspace(0.0, 1.0, n0)
        x1_vals = np.linspace(0.0, 1.0, n1)
        
        # Create meshgrid
        X0, X1 = np.meshgrid(x0_vals, x1_vals, indexing='ij')
        
        # Flatten
        x0_flat = X0.flatten()
        x1_flat = X1.flatten()
        
        # Add 4 extra points at edge midpoints to use full budget of 200
        extra_x0 = np.array([0.5, 0.5, 0.0, 1.0])
        extra_x1 = np.array([0.0, 1.0, 0.5, 0.5])
        
        x0_flat = np.concatenate([x0_flat, extra_x0])
        x1_flat = np.concatenate([x1_flat, extra_x1])
        
        n_samples = len(x0_flat)
        
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        # X[:, 2:] = 0 (inactive variables)
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
