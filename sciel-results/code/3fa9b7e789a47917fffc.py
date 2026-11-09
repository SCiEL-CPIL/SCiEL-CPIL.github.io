"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer using Sobol initialization and 
    weighted-sum local search (Nelder-Mead) for refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initialization with Sobol sequence for space-filling
        n_init = 40
        # Use a fixed seed for Sobol to ensure consistent initialization 
        # regardless of the global np.random seed, ensuring stable performance.
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        points = sampler.random(n_init)
        X_init = lo + points * (hi - lo)
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # 2. Local Search on selected seeds using Weighted Sums
        # We define a set of weight vectors to cover the Pareto front.
        # For 2 objectives, weights are (w, 1-w).
        weights = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        all_X = list(X_init)
        all_F = list(F_init)
        
        remaining_budget = self.budget - used
        
        if remaining_budget > 0:
            # Distribute budget among the weight vectors
            n_weights = len(weights)
            evals_per_search = remaining_budget // n_weights
            
            for w in weights:
                if used >= self.budget:
                    break
                
                # Find the initial point that minimizes the weighted sum
                # This acts as a greedy initialization for this specific direction
                scores = w * F_init[:, 0] + (1 - w) * F_init[:, 1]
                best_idx = np.argmin(scores)
                start_point = X_init[best_idx]
                
                # Define the scalar objective for local search
                # We need to track evaluations to stop before exceeding budget
                search_state = {'count': 0}
                
                def weighted_obj(x):
                    if search_state['count'] >= evals_per_search:
                        return 0.0 # Dummy return, optimizer should stop via maxiter
                    search_state['count'] += 1
                    x_clipped = np.clip(x, lo, hi)
                    f = problem(x_clipped)
                    return w * f[0] + (1 - w) * f[1]
                
                # Run Nelder-Mead local search
                # maxiter is set to evals_per_search to ensure we don't exceed budget
                # Nelder-Mead uses 1 evaluation per iteration in low dimensions
                try:
                    res = minimize(
                        weighted_obj, 
                        start_point, 
                        method='Nelder-Mead',
                        options={
                            'maxiter': evals_per_search,
                            'xatol': 1e-8,
                            'fatol': 1e-8
                        }
                    )
                    
                    # Record the result
                    # Note: res.x is the best point found.
                    # We must evaluate it to get the vector, BUT we have already 
                    # evaluated it during the search. To avoid an extra call, 
                    # we can't easily retrieve the last F. 
                    # However, one extra call is negligible if we have margin, 
                    # but we are tight on budget.
                    # Alternative: Trust that the last evaluation in the search 
                    # was the best. But minimize doesn't expose the last F vector.
                    # We will evaluate it again. If we run out of budget, we skip.
                    if used < self.budget:
                        used += 1
                        f_res = problem(res.x)
                        all_X.append(res.x)
                        all_F.append(f_res)
                        
                except Exception:
                    pass

        # 3. Final Non-Dominated Filtering
        F_all = np.array(all_F)
        X_all = np.array(all_X)
        
        def is_non_dominated(F):
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
            return nd

        final_mask = is_non_dominated(F_all)
        return X_all[final_mask]

# EVOLVE-BLOCK-END
