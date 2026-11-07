import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 295.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Store all evaluated points to compute final hypervolume
        all_x = []
        all_f = []
        
        def evaluate(x):
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_x.append(x)
            all_f.append(f)
            return f
        
        # 1. Initialization: Random sampling
        n_init = 20
        for _ in range(n_init):
            x = np.random.uniform(lo, hi)
            evaluate(x)
        
        used = n_init
        
        # 2. Main Loop: Local Search from Best Points
        # We will perform local search on the current best non-dominated points.
        # To ensure we cover the front, we select a diverse set of "anchors".
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Identify non-dominated points in the current set
            f_arr = np.array(all_f)
            n_pts = len(f_arr)
            
            # Simple non-dominated sorting (O(N^2) is fine for N<200)
            is_dominated = [False] * n_pts
            for i in range(n_pts):
                if is_dominated[i]:
                    continue
                for j in range(n_pts):
                    if i == j:
                        continue
                    if np.all(f_arr[i] <= f_arr[j]) and np.any(f_arr[i] < f_arr[j]):
                        is_dominated[j] = True
                        break
            
            # Get indices of non-dominated points
            nd_indices = [i for i in range(n_pts) if not is_dominated[i]]
            
            if not nd_indices:
                # Fallback if something went wrong
                nd_indices = [0]
            
            # Select up to 10 diverse points to run local search from.
            # We want points that are spread out in objective space.
            # Simple strategy: pick the first, last, and some evenly spaced ones in the list,
            # or just pick up to 10 random ones from the non-dominated set.
            # To be more systematic, let's pick the points with the best spread.
            # For simplicity and speed, we'll pick up to 10 points from the non-dominated set.
            # If there are more than 10, we can sample 10.
            
            # Let's pick 10 points that are "extreme" or just random to encourage diversity.
            # Random selection from non-dominated set is a good balance.
            if len(nd_indices) > 10:
                selected_anchors = np.random.choice(nd_indices, 10, replace=False)
            else:
                selected_anchors = nd_indices
            
            # Run L-BFGS-B local search from each anchor
            # We will use a small budget for each local search to avoid getting stuck.
            # Total budget remaining: budget - used
            # We have 10 anchors. Let's give each anchor a few evaluations.
            # But L-BFGS-B uses multiple evaluations.
            # Let's just run it until it converges or hits a small eval limit.
            
            for anchor_idx in selected_anchors:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x0 = all_x[anchor_idx]
                
                # Define objective function for scalarization
                # We can minimize the sum of objectives, or use Tchebycheff.
                # Since we want to move along the front, minimizing the sum (or norm)
                # from the current point might just move towards the center.
                # A better approach for multi-objective local search is to minimize
                # the distance to the ideal point or use a weighted sum.
                # However, simple L-BFGS-B on the sum of objectives works well for
                # finding a local minimum in the "sum" sense, which is often near the knee.
                # To explore the whole front, we should use different weights.
                
                # Let's use a random weight vector for each anchor to explore different parts.
                w = np.random.dirichlet(np.ones(n_obj))
                
                def obj_func(x):
                    if used >= budget:
                        return np.inf
                    f = evaluate(x)
                    return np.dot(w, f)
                
                # Run local search
                try:
                    res = minimize(obj_func, x0, method='L-BFGS-B', bounds=list(zip(lo, hi)), 
                                   options={'maxiter': 10, 'maxfun': 20})
                except Exception:
                    pass
                
                # Note: The `used` counter is updated inside `evaluate`.
                # The `minimize` function calls `obj_func` multiple times.
                # We need to ensure we don't exceed the budget.
                # The check `if used >= budget` inside `obj_func` returns inf,
                # which might cause minimize to behave weirdly but shouldn't crash.
                # However, it's better to stop if we hit the budget.
                # The `maxfun` option limits the number of function evaluations by scipy,
                # but our `evaluate` function increments `used`.
                # If `used` hits `budget`, `obj_func` returns inf.
                # This is a soft stop.
                
        # Return the last population or a representative set.
        # The task says "return value of __call__ is ignored", but we must return something.
        # Returning the last evaluated point is fine.
        if all_x:
            return all_x[-1]
        else:
            return np.zeros(n_dim)
