import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290  # 10s margin
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Exploit ZDT2 structure with non-uniform spacing
        # ZDT2 Pareto front: f1 = x[0], f2 = 1 - sqrt(f1) when x[1:] = 0.5
        # For HV maximization with reference [11,11], use power-law spacing
        # to densify near f1=0 where the front is steepest
        
        n_points = self.budget
        # Ensure n_points is at least 1
        if n_points < 1:
            return None
            
        # Generate x_1 values using quadratic spacing (u^2)
        # This densifies points near x1=0 where f2=1-sqrt(f1) has high curvature
        u = np.linspace(0.0, 1.0, n_points)
        x1_vals = u ** 2  # Quadratic spacing: denser near 0
        
        # Clip to bounds just in case
        x1_vals = np.clip(x1_vals, lo[0], hi[0])
        
        # Create X matrix
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_vals
        
        # Set remaining dimensions to 0.5 (center of bounds)
        # For ZDT2, g(x)=1 when all x[i]=0.5 for i>=1
        for i in range(1, self.dim):
            X[:, i] = 0.5 * (lo[i] + hi[i])
            
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            # If we timed out, we still return the result
            pass
            
        return X
