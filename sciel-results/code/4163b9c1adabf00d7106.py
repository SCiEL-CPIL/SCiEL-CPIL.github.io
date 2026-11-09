import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.func = func
        self.deadline = self.start_time + 290.0
        self.best_x = None
        self.best_f = float('inf')

        def wrapped_func(x):
            if self.call_count >= self.budget:
                return float('inf')
            if time.time() > self.deadline:
                return float('inf')
            
            x = np.asarray(x)
            if x.shape != (self.dim,):
                x = np.reshape(x, (self.dim,))
            
            self.call_count += 1
            f_val = float(func(x))
            
            if np.isfinite(f_val) and f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            
            return f_val

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Global Scan with Latin Hypercube Sampling (LHS)
        # Allocate 40% of budget to this phase to ensure rapid early AOCC improvement.
        # 200 evaluations for dim=5 is sufficient to find a decent basin in Gallagher 101.
        n_scan = int(budget * 0.4)
        # Ensure n_scan is at least 10
        n_scan = max(10, n_scan)
        
        # Generate LHS points
        # LHS ensures uniform coverage of the space, which is crucial for multimodal functions.
        points = np.zeros((n_scan, dim))
        for d in range(dim):
            perm = rng.permutation(n_scan)
            points[:, d] = (perm + rng.random(n_scan)) / n_scan
            points[:, d] = points[:, d] * 10.0 - 5.0 # Scale to [-5, 5]

        # Evaluate all scan points
        scan_fitness = np.full(n_scan, float('inf'))
        for i in range(n_scan):
            if self.call_count >= budget or time.time() > self.deadline:
                break
            scan_fitness[i] = wrapped_func(points[i])
        
        # Identify the best point from the scan
        if self.call_count < budget and self.best_x is not None:
            # Sort scan results to find the best
            valid_idx = np.where(np.isfinite(scan_fitness))[0]
            if len(valid_idx) > 0:
                best_scan_idx = valid_idx[np.argmin(scan_fitness[valid_idx])]
                # We already have self.best_x updated by wrapped_func, 
                # but let's ensure we use the specific best scan point for local search
                # Actually, wrapped_func updates self.best_x, so self.best_x is already the global best so far.
                # However, we might want to run local search on the *best of the scan*, 
                # which is self.best_x if it came from the scan.
                
                # Phase 2: Local Refinement using Nelder-Mead
                # Allocate the remaining budget to Nelder-Mead.
                # Nelder-Mead is robust and requires no gradient.
                # It works well for smooth basins around the initial point.
                # We will run it for the remaining budget, or until convergence.
                
                remaining_budget = budget - self.call_count
                if remaining_budget > 0 and self.best_x is not None:
                    # Nelder-Mead parameters
                    # maxiter is set high, but we rely on budget/time limits via wrapped_func
                    # xatol and fatol are set tight to ensure convergence if possible
                    try:
                        res = minimize(
                            wrapped_func, 
                            self.best_x, 
                            method='Nelder-Mead',
                            options={
                                'maxiter': remaining_budget, # Upper bound, actual limit is budget
                                'xatol': 1e-8,
                                'fatol': 1e-8,
                                'adaptive': True # Adaptive Nelder-Mead is more robust
                            }
                        )
                        # wrapped_func already updated self.best_x and self.best_f
                    except Exception:
                        pass

        # Fallback if no calls were made (should not happen)
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            if self.call_count < budget:
                self.best_f = wrapped_func(self.best_x)
            else:
                self.best_f = 0.0

        return self.best_x, self.best_f
