"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for DTLZ4.
    Strategy:
    1. Identify optimal position variables (distance to front) via local optimization.
    2. Sample the Pareto front geometry using Sobol sequences for shape variables.
    3. Evaluate combined points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 250.0  # Safety margin from 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_shape = self.n_obj - 1
        n_pos = self.dim - n_shape
        
        # 1. Optimize position variables
        # Fix shape variables to 0.5 (mid-range)
        shape_init = 0.5 * (hi[:n_shape] - lo[:n_shape]) + lo[:n_shape]
        
        def obj_pos(x_pos):
            x = np.empty(self.dim)
            x[:n_shape] = shape_init
            x[n_shape:] = x_pos
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            f = problem(x)
            # Minimize the sum of objectives to find the front (g(x) min)
            return np.sum(f)
        
        # Initial guess for position variables
        x_pos_init = 0.5 * (hi[n_shape:] - lo[n_shape:]) + lo[n_shape:]
        
        # Use Nelder-Mead to find optimal position
        # Limit iterations to save time
        try:
            res = minimize(obj_pos, x_pos_init, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
            x_pos_opt = res.x
            evals_used = res.nfev
        except Exception:
            x_pos_opt = x_pos_init
            evals_used = 10 # Estimate
        
        # Check time
        if time.time() - self.start_time > deadline:
            return []

        # 2. Sample the Pareto Front
        # Remaining budget
        remaining_budget = self.budget - evals_used
        if remaining_budget < 10:
            # Fallback: just evaluate the init point
            x = np.empty(self.dim)
            x[:n_shape] = shape_init
            x[n_shape:] = x_pos_opt
            problem(x)
            return []

        # Use Sobol for shape variables
        # Find largest power of 2 <= remaining_budget
        n_samples = 1
        while n_samples * 2 <= remaining_budget:
            n_samples *= 2
        
        sampler = qmc.Sobol(d=n_shape, scramble=True)
        shape_samples = sampler.random(n_samples)
        
        # Map to bounds
        shape_samples = shape_samples * (hi[:n_shape] - lo[:n_shape]) + lo[:n_shape]
        
        # Evaluate points
        for i in range(n_samples):
            if time.time() - self.start_time > deadline:
                break
            
            x = np.empty(self.dim)
            x[:n_shape] = shape_samples[i]
            x[n_shape:] = x_pos_opt
            x = np.clip(x, lo, hi)
            problem(x)
            
        return []
# EVOLVE-BLOCK-END
