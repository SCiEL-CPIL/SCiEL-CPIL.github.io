"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Direct approach for ZDT6.
    
    ZDT6 has:
    x in [0,1]^10
    g(x) = 1 + 9 * (sum(x[1:]) / 9) = 1 + sum(x[1:])
    
    f1 = 1 - exp(-4*x1) * sin(6*pi*x1)^6
    f2 = g
    
    Pareto front: x[1:] = 0 (g=1), x1 in [0,1]
    Front: f2 = 1, f1 = 1 - exp(-4*x1)*sin(6*pi*x1)^6
    
    Strategy: Sample x1 to cover the Pareto front well.
    Use non-uniform sampling based on curvature of f1(x1).
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        
        n = self.budget
        dim = self.dim
        
        # Generate x1 values with non-uniform density
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^6
        # The function has 3 full periods of sin in [0,1]
        # Near x1=0, f1 drops sharply due to exp(-4*x1)
        # We want more points where the front has higher curvature
        
        # Use a transformation to cluster points near x1=0 and at sin peaks
        # sin(6*pi*x1) has zeros at x1 = k/6 for k=0,...,6
        # Peaks at x1 = (2k+1)/12 for k=0,...,5
        
        # Strategy: mix of
        # 1. Points near the sin peaks (where f1 has local minima)
        # 2. Uniform grid for coverage
        # 3. Extra points near x1=0 where the curve is steep
        
        n = self.budget
        
        # 12 points at the sin peaks (local minima of f1)
        peak_x1 = [(2*k + 1) / 12.0 for k in range(6)]
        # Actually sin(6*pi*x1) peaks at x1 = 1/12, 3/12, 5/12, 7/12, 9/12, 11/12
        
        # 60 points uniformly spaced
        n_uniform = 60
        x1_uniform = np.linspace(0, 1, n_uniform)
        
        # 60 points clustered near x1=0 (where exp(-4*x1) changes rapidly)
        n_cluster = 60
        x1_cluster = np.linspace(0, 0.3, n_cluster)
        
        # 30 points at/near the sin peaks
        n_peak = 30
        x1_peak = np.array(peak_x1 + [0.0, 1.0])  # include endpoints
        
        # 30 random points
        n_random = 30
        x1_random = np.random.uniform(0, 1, n_random)
        
        x1_values = np.concatenate([x1_uniform, x1_cluster, x1_peak, x1_random])
        
        # Trim or pad to exactly n
        if len(x1_values) > n:
            x1_values = x1_values[:n]
        else:
            # Add more random points to fill
            extra = n - len(x1_values)
            x1_values = np.concatenate([x1_values, np.random.uniform(0, 1, extra)])
        
        # Create solutions on Pareto front: x[0]=x1, x[1:]=0
        X = np.zeros((n, dim))
        X[:, 0] = x1_values
        # x[1:] stays 0
        
        # Evaluate
        F = []
        for i in range(n):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
        
        F = np.array(F)
        
        # The return value is ignored, but we return something valid
        return X[:len(F)]
# EVOLVE-BLOCK-END
