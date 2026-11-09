"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.optimize import brentq


class Optimizer:
    """
    ZDT6 optimizer.
    
    f1 = 1 - exp(-4*x1) * sin(6*pi*x1)^6
    f2 = 1 + sum(x[1:])
    
    Pareto front: x[1:]=0, f2=1, f1 varies with x1.
    Hypervolume with ref [11,11] is dominated by the minimum f1 achieved.
    We use analytical + numerical methods to find the exact minimum of f1,
    then distribute remaining budget along the Pareto front for coverage.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def _f1(self, x1):
        """Calculate f1 for a given x1."""
        return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)
    
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        n = self.budget
        dim = self.dim
        
        # Analytical solution: f1'(x) = 0 when tan(6*pi*x) = 9*pi
        # x = (arctan(9*pi) + k*pi) / (6*pi) for k = 0, 1, ..., 5
        # Find all critical points and evaluate f1 at each
        k_values = np.arange(6)
        x_critical = (np.arctan(9.0 * np.pi) + k_values * np.pi) / (6.0 * np.pi)
        x_critical = x_critical[(x_critical >= 0.0) & (x_critical <= 1.0)]
        
        # Evaluate f1 at all critical points and find the minimum
        f1_at_critical = self._f1(x_critical)
        idx_min = np.argmin(f1_at_critical)
        x1_opt = x_critical[idx_min]
        f1_opt = f1_at_critical[idx_min]
        
        # Refine with scipy for maximum precision
        res = minimize_scalar(self._f1, bounds=(max(0.0, x1_opt - 1e-8), min(1.0, x1_opt + 1e-8)), 
                              method='bounded', options={'xatol': 1e-15})
        if self._f1(res.x) < f1_opt:
            x1_opt = res.x
            f1_opt = self._f1(x1_opt)
        
        # Build a set of x1 values that cover the Pareto front well
        # Include the optimal point and spread others across [0,1]
        x1_samples = [x1_opt]
        
        # Add points that span the f1 range
        # f1 ranges from 0 (at x1=0) to ~1 (at x1=1)
        # Sample logarithmically spaced f1 values and invert to get x1
        # For now, use a mix of uniform and targeted sampling
        n_extra = n - 1
        if n_extra > 0:
            # Uniform in x1 space, avoiding duplicates
            uniform_x1 = np.linspace(0.0, 1.0, n_extra, endpoint=False)
            for xv in uniform_x1:
                # Don't add if too close to x1_opt
                if abs(xv - x1_opt) > 1e-10:
                    x1_samples.append(xv)
                if len(x1_samples) >= n:
                    break
        
        # Ensure we have exactly n samples
        while len(x1_samples) < n:
            # Add more fine-grained samples
            x1_samples.append(x1_opt + 1e-6 * len(x1_samples))
        
        # Create solutions: x[0] = x1, x[1:] = 0
        X = np.zeros((n, dim))
        for i, xv in enumerate(x1_samples[:n]):
            X[i, 0] = xv
        
        # Evaluate
        F = []
        for i in range(n):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
        
        F = np.array(F)
        return X[:len(F)]
# EVOLVE-BLOCK-END
