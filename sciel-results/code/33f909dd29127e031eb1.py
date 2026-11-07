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

        # 1. Halton Sequence Initialization
        # Better coverage than random for low dimensions
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(n, dim):
            p = np.zeros(dim)
            primes = [2, 3, 5, 7]
            for d in range(dim):
                p[d] = halton_sequence(n, primes[d])
            return p

        n_init = 25
        for i in range(1, n_init + 1):
            if used >= budget or time.time() > deadline:
                break
            p = halton_point(i, dim)
            x = lo + (hi - lo) * p
            evaluate(x)

        if used >= budget or time.time() > deadline:
            return all_X

        # 2. Decomposition via Weighted Sum
        # Include extreme points to capture the ends of the Pareto front
        weights = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.0, 1.0]
        ])
        
        # Normalize weights to sum to 1? 
        # In weighted sum minimization, scaling doesn't change the argmin, 
        # but it affects the step size in some optimizers. 
        # Nelder-Mead is scale-invariant in terms of argmin, but 
        # let's keep them as is or normalize to sum=1 for consistency.
        # The function to minimize is w[0]*f0 + w[1]*f1.
        
        # Sort existing points to find best start for each weight
        F_arr = np.array(all_F)
        
        for w_vec in weights:
            if used >= budget or time.time() > deadline:
                break
            
            # Find best starting point for this weight
            if len(all_F) > 0:
                weighted_scores = F_arr[:, 0] * w_vec[0] + F_arr[:, 1] * w_vec[1]
                start_idx = np.argmin(weighted_scores)
                x_start = all_X[start_idx].copy()
            else:
                x_start = (lo + hi) / 2.0

            # Define objective function for this weight
            def weighted_sum(x, w=w_vec):
                if used >= budget or time.time() > deadline:
                    return np.inf
                f = evaluate(x)
                return w[0] * f[0] + w[1] * f[1]

            # Run Nelder-Mead
            stop_flag = [False]
            
            def callback(xk):
                nonlocal used
                if used >= budget - 2 or time.time() > deadline:
                    stop_flag[0] = True
                    return True
                return False

            try:
                # maxiter is a safety net, budget is the real constraint
                # xatol and fatol help stop early if converged
                res = minimize(
                    weighted_sum, 
                    x_start, 
                    method='Nelder-Mead',
                    options={'maxiter': 30, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True},
                    callback=callback
                )
            except Exception:
                pass

        return all_X
