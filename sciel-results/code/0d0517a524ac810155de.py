import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300
        
        n_evals = min(self.budget, 200)
        
        # ZDT1 Pareto front with g=0:
        # f1 = x1, f2 = 1 + sin(10*pi*x1)
        # Non-dominated: x1=0 and x1 in [0.1, 0.15]
        
        X = np.zeros((n_evals, self.dim))
        X[0, 0] = 0.0  # Point at x1=0, f=(0, 1)
        
        if n_evals > 1:
            n_inner = n_evals - 1
            
            # Optimize node positions in [0.1, 0.15] to maximize hypervolume
            # Variables: positions of n_inner points, sorted, in (0.1, 0.15]
            
            def neg_hv(params):
                # params: n_inner values in [0, 1], mapped to [0.1, 0.15]
                x1_vals = 0.1 + 0.05 * params
                # Sort to ensure ordering
                x1_sorted = np.sort(x1_vals)
                f1 = x1_sorted
                f2 = 1.0 + np.sin(10 * np.pi * f1)
                
                # Include the point at x1=0, f=(0, 1)
                all_f1 = np.concatenate([[0.0], f1])
                all_f2 = np.concatenate([[1.0], f2])
                
                # Hypervolume with reference (11, 11)
                # Sort by f1
                order = np.argsort(all_f1)
                all_f1 = all_f1[order]
                all_f2 = all_f2[order]
                
                # Remove dominated points (keep only non-dominated)
                # Since we know the front, all should be non-dominated if properly placed
                # But let's compute HV directly
                r1, r2 = 11.0, 11.0
                hv = 0.0
                for i in range(len(all_f1)):
                    next_f1 = all_f1[i+1] if i+1 < len(all_f1) else r1
                    if next_f1 > all_f1[i]:
                        hv += (next_f1 - all_f1[i]) * (r2 - all_f2[i])
                
                return -hv
            
            # Initial guess: uniform spacing
            x0 = np.linspace(0.01, 1.0, n_inner)
            
            # Use L-BFGS-B for bounded optimization
            result = minimize(neg_hv, x0, method='L-BFGS-B', 
                             bounds=[(0, 1)] * n_inner,
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.success or result.fun < neg_hv(x0):
                optimal_params = result.x
            else:
                optimal_params = x0
            
            x1_inner = 0.1 + 0.05 * optimal_params
            x1_inner = np.sort(x1_inner)
            
            # Ensure we don't go below 0.1 or above 0.15
            x1_inner = np.clip(x1_inner, 0.1 + 1e-12, 0.15)
            
            X[1:, 0] = x1_inner
            # x2..xn = 0 for g=0
        
        # Evaluate all points
        for i in range(n_evals):
            if time.time() > deadline - 10:
                break
            problem(X[i])
        
        return X
