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
        deadline = start_time + 280 # 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        eval_count = 0
        all_X = []
        all_F = []
        
        def safe_eval(x):
            nonlocal eval_count
            if eval_count >= self.budget or time.time() > deadline:
                return None
            x_c = np.clip(x, lo, hi)
            try:
                f = np.array(problem(x_c))
            except Exception:
                return None
            eval_count += 1
            all_X.append(x_c)
            all_F.append(f)
            return f

        # 1. Initial Sampling
        n_init = 20
        X_init = np.random.uniform(lo, hi, (n_init, self.dim))
        F_init = np.zeros((n_init, self.n_obj))
        valid_init = 0
        for i in range(n_init):
            f = safe_eval(X_init[i])
            if f is not None:
                F_init[valid_init] = f
                valid_init += 1
        
        if valid_init == 0:
            return all_X

        # 2. Weight Vectors
        # 3 objectives -> 2-simplex
        weights = []
        # Vertices
        weights.append([1.0, 0.0, 0.0])
        weights.append([0.0, 1.0, 0.0])
        weights.append([0.0, 0.0, 1.0])
        # Edge midpoints
        weights.append([0.5, 0.5, 0.0])
        weights.append([0.5, 0.0, 0.5])
        weights.append([0.0, 0.5, 0.5])
        # Center
        weights.append([1/3.0, 1/3.0, 1/3.0])
        # Additional
        weights.append([0.7, 0.2, 0.1])
        weights.append([0.2, 0.7, 0.1])
        weights.append([0.1, 0.2, 0.7])
        
        weights = np.array(weights)
        n_weights = len(weights)
        
        # 3. Local Search
        remaining = self.budget - eval_count
        max_evals_per_opt = max(1, int(remaining / n_weights))
        
        for w in weights:
            if time.time() > deadline or eval_count >= self.budget:
                break
            
            # Find best start from valid initial points
            # We need to use the valid part of F_init
            F_valid = F_init[:valid_init]
            X_valid = X_init[:valid_init]
            
            scores = F_valid @ w
            best_idx = np.argmin(scores)
            x0 = X_valid[best_idx]
            
            def scalar_obj(x):
                if time.time() > deadline:
                    return 1e10
                f = safe_eval(x)
                if f is None:
                    return 1e10
                return np.dot(w, f)
            
            try:
                # Nelder-Mead
                res = minimize(scalar_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': max_evals_per_opt, 
                                        'maxfev': max_evals_per_opt, 
                                        'xatol': 1e-5, 
                                        'fatol': 1e-5})
            except Exception:
                pass

        # 4. Final Polish (if budget remains)
        if eval_count < self.budget and time.time() < deadline:
            # Identify non-dominated points from all_F
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            
            # Simple non-dominated check
            # A point i is dominated if there exists j such that F_j <= F_i and F_j != F_i
            non_dom_mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                if not non_dom_mask[i]:
                    continue
                for j in range(len(F_arr)):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        non_dom_mask[i] = False
                        break
            
            nd_indices = np.where(non_dom_mask)[0]
            
            # Polish top non-dominated points
            # Use remaining budget for a few small local searches
            remaining = self.budget - eval_count
            if remaining > 0:
                # Pick up to 5 non-dominated points
                nd_indices = nd_indices[:min(5, len(nd_indices))]
                evals_per_polish = max(1, int(remaining / max(1, len(nd_indices))))
                
                for idx in nd_indices:
                    if time.time() > deadline or eval_count >= self.budget:
                        break
                    
                    x0 = X_arr[idx]
                    # Use a weighted sum where weights are inverse of objective values (normalized)
                    # Or just minimize sum of objectives? 
                    # Let's use a generic refinement: minimize sum of objectives
                    # But this might bias towards the center.
                    # Better: use the weight vector that this point best satisfies?
                    # Simpler: just run a small NM on the sum of objectives.
                    
                    def sum_obj(x):
                        if time.time() > deadline:
                            return 1e10
                        f = safe_eval(x)
                        if f is None:
                            return 1e10
                        return np.sum(f)
                        
                    try:
                        res = minimize(sum_obj, x0, method='Nelder-Mead', 
                                       options={'maxiter': evals_per_polish, 
                                                'maxfev': evals_per_polish, 
                                                'xatol': 1e-5, 
                                                'fatol': 1e-5})
                    except Exception:
                        pass

        return all_X
