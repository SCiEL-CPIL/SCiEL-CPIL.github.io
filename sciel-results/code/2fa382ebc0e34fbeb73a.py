import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # 40s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # --- Tchebycheff decomposition approach ---
        # Generate uniformly distributed weight vectors on the simplex
        # For 3 objectives, weights (w1, w2, w3) with w1+w2+w3=1, wi >= 0
        n_weights = 30
        weights = []
        # Use a grid approach for uniform distribution on the 2-simplex
        # w3 = 0, w1+w2=1: points (i/n, (n-i)/n, 0) for i=0..n
        # w3 = j/n, w1+w2=1-j/n: points on the line
        # For better uniformity, use a stratified approach
        for i in range(n_weights):
            # Use quasi-random or stratified sampling
            # Simple approach: divide the simplex into triangles
            # For 30 weights, use a 6x5 grid on the triangle
            pass
        
        # Better: use a systematic approach
        # Generate weights on a grid
        weights = []
        n_grid = 6  # 6 levels along each axis
        for i in range(n_grid + 1):
            for j in range(n_grid + 1 - i):
                w1 = i / n_grid
                w2 = j / n_grid
                w3 = 1.0 - w1 - w2
                if w3 >= -1e-10:
                    weights.append([w1, w2, max(0, w3)])
        
        # If we have more than n_weights, sample uniformly
        if len(weights) > n_weights:
            indices = np.random.choice(len(weights), n_weights, replace=False)
            weights = [weights[i] for i in indices]
        
        # Ensure we have exactly n_weights
        while len(weights) < n_weights:
            weights.append([1/3, 1/3, 1/3])
        
        weights = np.array(weights[:n_weights])
        
        # For each weight vector, solve a Tchebycheff sub-problem
        # Tchebycheff scalarization: min max_i (w_i * f_i(x))
        # This is equivalent to minimizing the maximum weighted deviation
        
        # Use Nelder-Mead for each sub-problem
        # Budget allocation: budget / n_weights evaluations per sub-problem
        evals_per_sub = max(1, budget // n_weights)
        
        # Store the best solution for each sub-problem
        best_solutions = []
        used = 0
        
        for idx, w in enumerate(weights):
            if time.time() > deadline:
                break
            if used >= budget:
                break
            
            # Initial point: center of the search space
            x0 = (lo + hi) / 2.0
            x0 = np.clip(x0, lo, hi)
            
            # Define the Tchebycheff objective
            def tcheb_obj(x, w=w):
                nonlocal used
                if used >= budget:
                    return 1e10
                used += 1
                f = problem(x)
                # Tchebycheff: max_i (w_i * f_i)
                return np.max(w * f)
            
            # Use Nelder-Mead
            from scipy.optimize import minimize
            try:
                result = minimize(tcheb_obj, x0, method='Nelder-Mead', 
                                  options={'maxiter': evals_per_sub, 'xatol': 1e-6, 'fatol': 1e-6})
                if result.success or result.fun < 1e9:
                    x_best = np.clip(result.x, lo, hi)
                    best_solutions.append(x_best)
            except Exception:
                # If Nelder-Mead fails, use a simple random search
                for _ in range(evals_per_sub):
                    if used >= budget:
                        break
                    x_rand = np.random.uniform(lo, hi)
                    x_rand = np.clip(x_rand, lo, hi)
                    f_rand = problem(x_rand)
                    tcheb_val = np.max(w * f_rand)
                    if len(best_solutions) == 0 or tcheb_val < np.max(w * problem(best_solutions[-1])):
                        best_solutions.append(x_rand)
                    used += 1
        
        # Collect all solutions and return
        if best_solutions:
            X = np.array(best_solutions)
        else:
            X = np.random.uniform(lo, hi, size=(1, n_dim))
        
        return X
