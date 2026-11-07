"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using Weighted Sum Decomposition with Local Nelder-Mead Search.
    This approach decomposes the multi-objective problem into a set of single-objective 
    sub-problems, each solved by a local optimization algorithm to ensure convergence 
    to the Pareto front, which is effective for smooth problems like RE21 with small budgets.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        # 1. Initial Random Sampling to identify promising regions
        # Use 10% of budget for initialization
        n_init = max(10, self.budget // 10)
        # Ensure we don't exceed budget
        if n_init > self.budget:
            n_init = self.budget
            
        # Generate initial random points
        # Use a fixed seed for reproducibility in the optimizer's internal logic?
        # The task says the evaluator sets np.random.seed(seed). 
        # We should use the global random state or a derived one.
        # Using np.random directly relies on the evaluator's seed.
        
        X_init = lo + (hi - lo) * np.random.random((n_init, dim))
        F_init = np.array([problem(x) for x in X_init])
        
        used_evals = n_init
        all_X = list(X_init)
        all_F = list(F_init)

        # 2. Decomposition via Weighted Sum
        # We define a set of weight vectors to cover the Pareto front.
        # For 2 objectives, weights are (w, 1-w).
        # We choose 5 weights to spread the effort.
        weights = [0.1, 0.3, 0.5, 0.7, 0.9]
        n_weights = len(weights)
        
        # Remaining budget for local searches
        remaining_budget = self.budget - used_evals
        if remaining_budget <= 0:
            return all_X
            
        # Evaluate each weight vector's objective on the initial points to find best starting point
        # This ensures the local search starts near the Pareto front for that specific trade-off
        
        # We will allocate the remaining budget equally among the weight vectors
        # But we must account for the overhead of the local optimizer.
        # Nelder-Mead in 4D with maxiter=30 usually uses ~30-40 function calls.
        
        budget_per_search = remaining_budget // n_weights
        
        for w1 in weights:
            if time.time() > deadline:
                break
                
            w2 = 1.0 - w1
            
            # Define the scalarized objective
            def scalar_obj(x):
                # Clip x to bounds to ensure validity, though minimizer should stay in bounds if we use bounds
                # But problem(x) clips internally. We should pass raw x or clipped?
                # The problem function clips. We pass x as is.
                f = problem(x)
                return w1 * f[0] + w2 * f[1]

            # Find the best initial point from X_init for this weight
            # Calculate weighted sum for all init points
            weighted_sums = w1 * F_init[:, 0] + w2 * F_init[:, 1]
            best_init_idx = np.argmin(weighted_sums)
            x0 = X_init[best_init_idx]
            
            # Run Nelder-Mead local search
            # maxiter controls the number of iterations. 
            # In scipy, maxiter is the number of iterations, not function calls.
            # However, function calls are roughly maxiter * (dim+1) or similar.
            # To strictly control function calls, we might need a custom wrapper or careful maxiter.
            # For dim=4, 1 iteration of NM does ~5 function calls.
            # So maxiter=10 -> ~50 calls.
            # We have budget_per_search. Let's set maxiter based on that.
            max_iter = max(5, budget_per_search // 5)
            
            try:
                res = minimize(scalar_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-6})
                
                # Record the result
                # Note: res.fun is the scalar objective, not the vector.
                # We need to evaluate the vector objectives for the final point.
                # However, the minimizer has already called problem() multiple times.
                # The last call in res.x might not have been explicitly recorded in our all_F if we didn't wrap it.
                # Wait, `problem` is called inside `scalar_obj`. 
                # The task says "Every returned objective vector is recorded".
                # The evaluator records the return value of `problem(x)`.
                # So we don't need to manually record them, the evaluator does.
                # We just need to ensure we don't exceed the budget.
                
                # Add the final point to our collection (it's already evaluated by the optimizer)
                # But we should check if it's better than what we have.
                # Since the evaluator records everything, we just need to ensure we stop.
                
                # To be safe and ensure we have the vector, we can evaluate it once more?
                # No, that wastes budget. The minimizer's internal calls are counted.
                # The final point res.x is the best found.
                # We can add it to all_X/all_F for internal tracking, but the evaluator 
                # has already recorded the vector from the last call to problem(res.x) 
                # IF the minimizer called it at the end. 
                # SciPy's Nelder-Mead does not necessarily call the function one last time 
                # at the exact res.x if it converged. It returns the best point found.
                # The evaluator records EVERY call. So all intermediate points are recorded.
                # The final point res.x might not have been evaluated as the very last call 
                # if the algorithm stopped due to tolerance.
                # However, the hypervolume is calculated on ALL recorded points.
                # So as long as we generate good points, we are fine.
                
            except Exception:
                pass

        return all_X
# EVOLVE-BLOCK-END
