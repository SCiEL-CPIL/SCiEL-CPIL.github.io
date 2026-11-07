import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # 1. Initialization
        # Assume first 3 variables are position, rest are distance.
        # Set distance variables to mid-range.
        n_pos = 3
        n_dist = dim - n_pos
        
        # Grid for position variables
        # Let's use a 4x4x4 grid = 64 points
        n_init = 64
        g = np.linspace(0, 1, 4)
        
        X_init = np.zeros((n_init, dim))
        count = 0
        for a in g:
            for b in g:
                for c in g:
                    if count >= n_init:
                        break
                    x = np.zeros(dim)
                    x[0] = lo[0] + a * (hi[0] - lo[0])
                    x[1] = lo[1] + b * (hi[1] - lo[1])
                    x[2] = lo[2] + c * (hi[2] - lo[2])
                    # Mid-range for distance variables
                    x[3:] = (lo[3:] + hi[3:]) / 2.0
                    X_init[count] = x
                    count += 1
        
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # 2. Selection
        # Find non-dominated points
        def is_non_dominated(F, idx):
            for j in range(len(F)):
                if j == idx:
                    continue
                if np.all(F[j] <= F[idx]) and np.any(F[j] < F[idx]):
                    return False
            return True
        
        nondom_indices = [i for i in range(len(F_init)) if is_non_dominated(F_init, i)]
        
        # Select up to 10 points for refinement
        # Sort by first objective and pick evenly
        if len(nondom_indices) > 10:
            sorted_idx = np.argsort(F_init[nondom_indices, 0])
            step = max(1, len(sorted_idx) // 10)
            selected = sorted_idx[::step][:10]
            search_indices = [nondom_indices[i] for i in selected]
        else:
            search_indices = nondom_indices
            
        remaining_budget = budget - used
        if remaining_budget <= 0 or len(search_indices) == 0:
            return X_init
        
        n_search = len(search_indices)
        evals_per_search = max(10, remaining_budget // n_search)
        
        # 3. Refinement
        # Use weighted sum minimization with different weights for each point
        # Generate weight vectors
        # For 3 objectives, use a set of weights
        # We can use a simplex lattice or just random weights
        # Let's use 10 random weights on the simplex
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in refinement weights
        weights = rng.dirichlet(np.ones(n_obj), size=n_search)
        
        for i, idx in enumerate(search_indices):
            if time.time() > deadline:
                break
            if used >= budget:
                break
                
            x0 = X_init[idx].copy()
            w = weights[i]
            
            def obj_func(x):
                nonlocal used
                if used >= budget:
                    return np.sum(F_init[idx])
                if time.time() > deadline:
                    return np.sum(F_init[idx])
                
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                used += 1
                return np.dot(w, f)
                
            try:
                res = minimize(obj_func, x0, method='Nelder-Mead', 
                               options={'maxiter': evals_per_search, 
                                        'xatol': 1e-5, 
                                        'fatol': 1e-5})
                X_init[idx] = res.x
                if used < budget and time.time() < deadline:
                    F_init[idx] = problem(res.x)
                    used += 1
            except:
                pass
        
        return X_init
