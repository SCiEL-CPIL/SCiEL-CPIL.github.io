import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.evals = 0
        self.start_time = None
        self.timeout = 320.0  # Safety margin below 330s

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_X = []
        all_F = []
        
        def safe_eval(x):
            if time.time() - self.start_time > self.timeout:
                raise TimeoutError("Time limit exceeded")
            if self.evals >= self.budget:
                raise ValueError("Budget exceeded")
            
            x = np.asarray(x)
            fx = problem(x)
            self.evals += 1
            return np.array(fx)

        # 1. Initial Random Sampling
        # Use 50% of budget for exploration
        n_rand = self.budget // 2
        if n_rand < 10:
            n_rand = 10
            
        for _ in range(n_rand):
            if time.time() - self.start_time > self.timeout:
                break
            x = np.random.uniform(lo, hi)
            try:
                fx = safe_eval(x)
                all_X.append(x)
                all_F.append(fx)
            except (TimeoutError, ValueError):
                break
        
        if not all_F:
            return np.empty((0, self.dim))

        F_arr = np.array(all_F)
        X_arr = np.array(all_X)
        
        # 2. Identify Non-Dominated Points
        def is_dominated(f, F):
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
        
        nondom_mask = np.array([not is_dominated(f, F_arr) for f in F_arr])
        start_pool = X_arr[nondom_mask] if np.any(nondom_mask) else X_arr
        start_pool_F = F_arr[nondom_mask] if np.any(nondom_mask) else F_arr
        
        # 3. Generate Diverse Weight Vectors
        # We want to cover the Pareto front. For 3 objectives, we generate points on the simplex.
        # Let's use a set of predefined diverse weights plus some random ones.
        n_weights = 12
        weights = []
        
        # Add extreme points
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)
        
        # Add uniform point
        weights.append(np.ones(self.n_obj) / self.n_obj)
        
        # Add random points on the simplex
        while len(weights) < n_weights:
            w = np.random.rand(self.n_obj)
            w /= np.sum(w)
            weights.append(w)
            
        # 4. Local Optimization with Scalarization
        # We have budget - n_rand evaluations left.
        remaining_budget = self.budget - self.evals
        if remaining_budget <= 0:
            return np.array(all_X)
            
        # Allocate remaining budget to local optimizations
        # Each optimization gets a limited number of function calls
        max_fun_per_opt = max(2, remaining_budget // n_weights)
        
        for w in weights:
            if self.evals >= self.budget:
                break
            if time.time() - self.start_time > self.timeout:
                break
            
            # Find the best starting point for this weight
            if len(start_pool) > 0:
                weighted_sums = start_pool_F @ w
                best_idx = np.argmin(weighted_sums)
                x0 = start_pool[best_idx]
            else:
                x0 = np.random.uniform(lo, hi)
                
            # Define scalar objective
            def scalar_obj(x, w=w):
                try:
                    fx = safe_eval(x)
                    return np.dot(fx, w)
                except (TimeoutError, ValueError):
                    return np.inf
            
            # Optimize
            try:
                res = minimize(scalar_obj, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 50, 'maxfun': max_fun_per_opt})
            except Exception:
                pass
                
        # 5. Fill remaining budget with random sampling if any left
        while self.evals < self.budget:
            if time.time() - self.start_time > self.timeout:
                break
            x = np.random.uniform(lo, hi)
            try:
                fx = safe_eval(x)
                all_X.append(x)
                all_F.append(fx)
            except (TimeoutError, ValueError):
                break
                
        if all_X:
            return np.array(all_X)
        else:
            return np.empty((0, self.dim))
