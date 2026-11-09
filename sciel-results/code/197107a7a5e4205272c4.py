import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective search: Random initialization + Local Search (Nelder-Mead)
    on non-dominated points to maximize hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random
        dim, budget = self.dim, self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        deadline = time.time() + 300.0
        safe = deadline - 30.0
        
        used_evals = 0
        
        # Phase 1: Random Sampling
        # Use 50% of budget for random sampling
        n_init = budget // 2
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        F_init = np.empty((n_init, self.n_obj))
        for i in range(n_init):
            F_init[i] = problem(X_init[i])
            used_evals += 1
            if used_evals >= budget:
                break
            if time.time() > safe:
                break
        
        # Find non-dominated set
        def find_non_dominated(F):
            n = len(F)
            if n == 0:
                return []
            nd = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        nd[i] += 1
            return [i for i in range(n) if nd[i] == 0]
            
        nd_indices = find_non_dominated(F_init)
        
        if not nd_indices:
            # Fallback: use all points
            nd_indices = list(range(n_init))
            
        # Sort non-dominated points by sum of objectives (proxy for quality)
        nd_points = [(F_init[i], X_init[i], i) for i in nd_indices]
        nd_points.sort(key=lambda x: np.sum(x[0]))
        
        # Phase 2: Local Search
        remaining_budget = budget - used_evals
        
        # Determine how many points to refine
        # Each NM run with maxiter=1 uses ~dim+1 = 6 evals
        evals_per_refine = dim + 1
        n_refine = remaining_budget // evals_per_refine
        
        if n_refine == 0:
            n_refine = 1
            
        # Take the best n_refine non-dominated points
        to_refine = nd_points[:min(n_refine, len(nd_points))]
        
        ref = np.array([1705.0, 12.0, 0.3])
        
        for f_val, x_val, idx in to_refine:
            if time.time() > safe:
                break
            if used_evals >= budget:
                break
            
            # Define local search objective: Minimize distance to reference
            def local_obj(x):
                nonlocal used_evals
                if used_evals >= budget:
                    return 1e10
                fx = problem(x)
                used_evals += 1
                # If not better than reference, return large value
                if not np.all(fx < ref):
                    return 1e10 + np.sum(fx)
                return np.sum((fx - ref)**2)
                
            try:
                # maxiter=1 allows one full simplex update
                res = minimize(local_obj, x_val, method='Nelder-Mead', 
                               options={'maxiter': 1, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                pass
                
        return X_init
