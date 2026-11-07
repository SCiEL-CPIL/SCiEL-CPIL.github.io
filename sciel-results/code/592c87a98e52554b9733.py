"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based multi-objective optimizer with weighted-sum decomposition."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Count evaluations
        eval_count = [0]
        
        def count_eval(x):
            eval_count[0] += 1
            return problem(x)
        
        # Generate diverse weight vectors for weighted sum
        # Use simplex-lattice or random weights
        n_weights = 15
        weights = []
        # Add extreme points
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)
        # Add uniform random weights
        rng = np.random.default_rng(42)
        for _ in range(n_weights - self.n_obj):
            w = rng.random(self.n_obj)
            w /= np.sum(w)
            weights.append(w)
        weights = np.array(weights)
        
        # Initial random points for seeding
        n_init = 5
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        F_init = np.array([count_eval(x) for x in X_init])
        
        # Store all evaluated points
        all_X = list(X_init)
        all_F = list(F_init)
        
        # For each weight vector, run L-BFGS-B to minimize weighted sum
        for w in weights:
            if time.time() > deadline:
                break
            if eval_count[0] >= self.budget:
                break
            
            # Choose best starting point based on current weighted sum
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                weighted_sums = F_arr @ w
                best_idx = np.argmin(weighted_sums)
                x0 = all_X[best_idx].copy()
            else:
                x0 = (lo + hi) / 2.0
            
            # Objective function for L-BFGS-B
            def obj(x):
                if eval_count[0] >= self.budget:
                    return np.sum(problem(x) * w)  # This won't be called if we check before
                fx = count_eval(x)
                return np.sum(fx * w)
            
            try:
                result = minimize(
                    obj, x0, method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 50, 'maxls': 20}
                )
                if result.success or result.fun < np.inf:
                    x_res = np.clip(result.x, lo, hi)
                    f_res = count_eval(x_res)
                    all_X.append(x_res)
                    all_F.append(f_res)
            except Exception:
                pass
        
        # If we have budget left, do some local refinement around best points
        if time.time() < deadline and eval_count[0] < self.budget:
            # Get non-dominated points
            if len(all_F) > 1:
                F_arr = np.array(all_F)
                X_arr = np.array(all_X)
                # Find non-dominated
                non_dom_mask = np.ones(len(F_arr), dtype=bool)
                for i in range(len(F_arr)):
                    for j in range(len(F_arr)):
                        if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                            non_dom_mask[i] = False
                            break
                nd_X = X_arr[non_dom_mask]
                nd_F = F_arr[non_dom_mask]
                
                # Do small perturbations around non-dominated points
                step_size = 0.05 * (hi - lo)
                for x_nd in nd_X[:5]:  # Limit to 5 points
                    if time.time() > deadline:
                        break
                    if eval_count[0] >= self.budget:
                        break
                    for _ in range(3):
                        if eval_count[0] >= self.budget:
                            break
                        x_new = x_nd + step_size * rng.standard_normal(self.dim)
                        x_new = np.clip(x_new, lo, hi)
                        f_new = count_eval(x_new)
                        all_X.append(x_new)
                        all_F.append(f_new)
        
        return all_X
# EVOLVE-BLOCK-END
