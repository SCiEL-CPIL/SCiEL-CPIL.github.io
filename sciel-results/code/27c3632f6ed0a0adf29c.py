"""Grid-based decomposition multi-objective optimizer for RE37."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Grid-based decomposition:
    1. Evaluate a coarse grid to map the objective landscape.
    2. Identify promising regions (non-dominated grid points).
    3. Use weighted-sum scalarization to guide local Nelder-Mead refinement
       from the best grid point for each weight vector, ensuring front coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # 1. Coarse Grid Evaluation
        # For dim=4, g=3 gives 81 points. This is a good balance.
        g = 3
        if budget < 100:
            g = 2 # 16 points, very coarse but safe for low budget
        
        grid_points_per_dim = g
        total_grid_points = grid_points_per_dim ** dim
        
        # Adjust g if budget is too small
        while total_grid_points > budget * 0.5 and grid_points_per_dim > 2:
            grid_points_per_dim -= 1
            total_grid_points = grid_points_per_dim ** dim
            
        # If grid is still too big, just use what fits
        if total_grid_points > budget:
             # Fallback: random sampling if grid is impossible
             # But for dim=4, g=2 is 16, g=3 is 81. Budget 200 is fine.
             pass

        axes = [np.linspace(lo[i], hi[i], grid_points_per_dim) for i in range(dim)]
        grid = np.array(np.meshgrid(*axes)).reshape(dim, -1).T
        
        # Evaluate grid
        F_grid = np.array([problem(x) for x in grid])
        used = total_grid_points

        # 2. Identify Non-Dominated Points
        def is_dominated(F):
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                fi = F[i]
                less_equal = np.all(F <= fi, axis=1)
                less_than = np.any(F < fi, axis=1)
                dominates = less_equal & less_than
                dominates[i] = False
                if np.any(dominates):
                    dominated[i] = True
            return dominated
        
        dom_grid = is_dominated(F_grid)
        non_dom_indices = np.where(~dom_grid)[0]
        
        if len(non_dom_indices) == 0:
            non_dom_indices = np.arange(len(grid))

        # 3. Targeted Local Refinement
        # We want to cover the front. We will use a set of weight vectors.
        # For n_obj=3, we can use a set of weights that span the simplex.
        # e.g., (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)
        # Normalized.
        
        weight_sets = [
            np.array([1.0, 0.1, 0.1]),
            np.array([0.1, 1.0, 0.1]),
            np.array([0.1, 0.1, 1.0]),
            np.array([1.0, 1.0, 0.1]),
            np.array([1.0, 0.1, 1.0]),
            np.array([0.1, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 0.5, 0.5]),
            np.array([0.5, 1.0, 0.5]),
            np.array([0.5, 0.5, 1.0]),
        ]
        
        # Normalize weights
        weight_sets = [w / np.sum(w) for w in weight_sets]
        
        remaining_budget = budget - used
        
        # Estimate cost per Nelder-Mead run. 
        # NM in 4D with maxfev=10-15 is typical.
        # Let's allocate budget dynamically.
        n_runs = len(weight_sets)
        # We can't run more than we have budget for.
        # Each run costs ~15 evals.
        max_runs_by_budget = max(1, remaining_budget // 15)
        n_runs = min(n_runs, max_runs_by_budget)
        
        # Select which weight sets to use if we can't use all
        # Prioritize extreme points first? Or just take the first N.
        # Taking first N ensures coverage of the "corners" and "center".
        
        for i in range(n_runs):
            if used >= budget:
                break
                
            w = weight_sets[i]
            
            # Find the grid point that minimizes the weighted sum
            # This is our starting point x0
            weighted_sums = F_grid @ w
            # We want the minimum weighted sum
            best_grid_idx = np.argmin(weighted_sums)
            x0 = grid[best_grid_idx].copy()
            
            # Define objective function for Nelder-Mead
            # We use the weighted sum as the scalar objective
            # But we must ensure we track evaluations correctly
            
            def obj(x, w=w):
                nonlocal used
                if used >= budget:
                    return np.inf
                fx = problem(x)
                used += 1
                return np.dot(w, fx)
            
            # Nelder-Mead settings
            # Small maxfev to save budget for other directions
            maxfev = min(15, budget - used)
            if maxfev <= 0:
                continue
                
            options = {
                'maxiter': 30,
                'maxfev': maxfev,
                'xatol': 1e-4,
                'fatol': 1e-6
            }
            
            try:
                res = minimize(obj, x0, method='Nelder-Mead', bounds=list(zip(lo, hi)), options=options)
                
                # If we have budget left, evaluate the final point explicitly
                # (minimize might stop before using all maxfev, or the last eval
                # in the loop might be the best)
                # Actually, minimize returns the best point found.
                # The evaluation of res.x is already counted in 'used' if it was
                # part of the iterations. However, to be safe and ensure the
                # specific best point is in the record, we might re-evaluate.
                # But re-evaluation costs 1 budget.
                # If used < budget, we can do a final check.
                if used < budget:
                    xf = np.clip(res.x, lo, hi)
                    # Check if this point is significantly better than the start?
                    # Or just record it.
                    ff = problem(xf)
                    used += 1
            except Exception:
                pass
        
        return grid
# EVOLVE-BLOCK-END
