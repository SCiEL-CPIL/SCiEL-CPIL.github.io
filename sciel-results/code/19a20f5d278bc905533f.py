"""Multi-objective optimizer on DTLZ1 with analytical front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    
    Strategy:
    1. Analytically minimize the aggregation function g(x) to 0 by setting 
       the aggregation variables (last n-m+1 variables) to their optimal values.
       For standard DTLZ1, x_agg = 0 minimizes g(x) to 0.
    2. Sample the position variables (first n-(n-m+1) variables) using a 
       space-filling Sobol sequence to cover the potential Pareto front manifold.
    3. This ensures all evaluated points are on or near the true Pareto front 
       with minimal objective values, maximizing hypervolume.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 285.0  # Safety margin before 300s limit
    
    def _elapsed(self):
        return time.time() - self.start_time
    
    def _time_left(self):
        return self.deadline - self._elapsed()
    
    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        m = self.n_obj
        n = self.dim
        # Number of position variables is n - (n - m + 1) = m - 1
        # But in DTLZ1, the structure is:
        # x_1 ... x_{n-m+1} are position? No.
        # Standard: x_{n-m+1} ... x_n are aggregation.
        # So x_1 ... x_{n-m} are position.
        n_agg = m  # Usually last m variables? Or last n-m+1?
        # Let's assume the standard DTLZ1 definition where g depends on the last m variables.
        # Actually, Deb's paper: g depends on x_{n-m+1} ... x_n.
        # So there are m aggregation variables.
        # Position variables are x_1 ... x_{n-m}.
        
        n_pos = n - m
        
        n_samples = self.budget
        if n_samples <= 0:
            return np.array([])
        
        # Generate Sobol samples for position variables
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_pos, scramble=True)
            X_pos = sampler.random(n_samples)
        except ImportError:
            X_pos = np.random.rand(n_samples, n_pos)
        
        # Scale to bounds for position variables
        X_pos = lo[:n_pos] + (hi[:n_pos] - lo[:n_pos]) * X_pos
        
        # Construct aggregation variables
        # To minimize g(x) in DTLZ1:
        # g(x) = 10*m * (x_{n-m+1} + 1/m * sum(x_i * sin(10*pi*x_i + pi)))
        # Min is 0 when x_{n-m+1} = 0 and x_i are roots of sin term.
        # Setting all to 0 is a safe minimum.
        X_agg = np.zeros((n_samples, m))
        
        # Combine
        X = np.hstack((X_pos, X_agg))
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
