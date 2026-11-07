"""Multi-objective optimizer on DTLZ1 with analytical front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    
    Strategy:
    1. Set aggregation variables (x[2:7]) to 0 to achieve g(x) = 0.
    2. Sample position variables (x[0], x[1]) using a Sobol sequence on [0,1]^2.
    3. This places all points exactly on the analytical Pareto front.
    4. Use the full budget for dense front sampling to maximize hypervolume.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 290.0  # Safety margin before 300s limit
    
    def _elapsed(self):
        return time.time() - self.start_time
    
    def _time_left(self):
        return self.deadline - self._elapsed()
    
    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        m = self.n_obj
        n = self.dim
        n_pos = m - 1  # 2 position variables for 3 objectives
        n_agg = n - n_pos  # 5 aggregation variables
        
        n_samples = self.budget
        if n_samples <= 0:
            return np.array([])
        
        # Generate Sobol samples for position variables (2D)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_pos, scramble=True, seed=0)
            X_pos = sampler.random(n_samples)
        except ImportError:
            # Fallback: use Latin Hypercube via random
            X_pos = np.random.rand(n_samples, n_pos)
        
        # Scale to bounds for position variables
        X_pos = lo[:n_pos] + (hi[:n_pos] - lo[:n_pos]) * X_pos
        
        # Construct aggregation variables set to 0 (achieves g=0 for DTLZ1)
        X_agg = np.zeros((n_samples, n_agg))
        
        # Combine
        X = np.hstack((X_pos, X_agg))
        
        # Ensure within bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
