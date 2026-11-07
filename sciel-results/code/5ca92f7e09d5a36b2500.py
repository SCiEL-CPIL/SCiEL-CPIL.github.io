import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 325.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        # 1. Initial Sampling
        n_init = 40
        for _ in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            evaluate(x)

        if used >= budget or time.time() > deadline:
            return all_X

        F_init = np.array(all_F)
        # Estimate ideal and nadir
        f_min = np.min(F_init, axis=0)
        f_max = np.max(F_init, axis=0)
        
        # 2. Decomposition via Weighted Sum
        # Weights for objective 1 (obj 2 weight is 1 - w)
        # Use a grid of weights
        n_weights = 10
        weights = np.linspace(0.05, 0.95, n_weights)
        
        # Remaining budget for local optimization
        remaining_budget = budget - used
        
        # Allocate budget per weight
        # Nelder-Mead in 4D: initial simplex is 5 points.
        # Each iteration is 1 eval (best move) + potentially 1 more (expansion/contraction).
        # Let's limit iterations to ensure we don't blow the budget.
        # 180 remaining / 10 weights = 18 evals per weight.
        # maxiter for NM is tricky, better to use a wrapper that counts evals.
        
        def weighted_sum(x, w):
            if used >= budget or time.time() > deadline:
                return np.inf
            f = evaluate(x)
            return w[0] * f[0] + w[1] * f[1]

        # For each weight, find a good starting point
        for w_val in weights:
            if used >= budget or time.time() > deadline:
                break
            
            w_vec = np.array([w_val, 1.0 - w_val])
            
            # Find starting point: min weighted sum among current all_F
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                weighted_scores = F_arr[:, 0] * w_vec[0] + F_arr[:, 1] * w_vec[1]
                start_idx = np.argmin(weighted_scores)
                x_start = all_X[start_idx].copy()
            else:
                x_start = (lo + hi) / 2.0

            # Run Nelder-Mead
            # We need to stop when budget is hit.
            # scipy minimize doesn't easily stop mid-run based on external counter without a callback.
            # We can use a callback.
            
            stop_flag = [False]
            
            def callback(xk):
                nonlocal used
                if used >= budget - 2 or time.time() > deadline:
                    stop_flag[0] = True
                    return True
                return False

            try:
                res = minimize(
                    weighted_sum, 
                    x_start, 
                    args=(w_vec,), 
                    method='Nelder-Mead',
                    options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True},
                    callback=callback
                )
            except Exception:
                pass

        return all_X
