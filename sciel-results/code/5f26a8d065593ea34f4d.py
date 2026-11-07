# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Grid-based decomposition with local Nelder-Mead simplex per cell."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # We will collect all evaluated points and their objective values
        all_X = []
        all_F = []

        def eval_point(x):
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        # Determine grid structure:
        # We want to partition the 4D space into hypercubes.
        # With 4 dimensions, a 2x2x2x2 grid gives 16 cells.
        # We have 200 evaluations. 16 cells * ~12 evaluations each = 192, leaving 8 for extra.
        # Let's use a 2x2x2x2 grid = 16 cells.
        grid_size = 2
        n_cells = grid_size ** dim  # 16 for dim=4

        # Points per cell for local optimization
        # Nelder-Mead simplex in 4D uses 5 initial points + iterations.
        # Let's allocate about 12 evaluations per cell: 5 for initial simplex + 7 iterations.
        # 16 * 12 = 192. Remaining 8 can be used for a second pass or extra in best cells.

        # For each cell, define its bounds
        cell_bounds = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        lo_c = lo + span * np.array([i, j, k, l]) / grid_size
                        hi_c = lo + span * np.array([i + 1, j + 1, k + 1, l + 1]) / grid_size
                        cell_bounds.append((lo_c, hi_c))

        # For each cell, run a local Nelder-Mead optimization on a scalarized objective.
        # To cover the Pareto front, we use different scalarization weights.
        # With 16 cells, we can assign different weight vectors to different cells.
        # Alternatively, for each cell, we can run multiple scalarizations.
        
        # Let's think differently: 
        # The directive says "treating each as a sub-problem to be optimized via local simplex search".
        # This suggests using Nelder-Mead (a simplex method) on each cell.
        # But Nelder-Mead minimizes a scalar function. For multi-objective, we need to scalarize.
        
        # Strategy: For each cell, pick a weight vector w, and minimize w·f(x) within the cell.
        # To cover the Pareto front, use different weight vectors.
        # With 16 cells and 16 different weight vectors, each cell gets one scalarization.
        
        # Generate 16 weight vectors uniformly distributed on the simplex {w_i >= 0, sum(w_i) = 1}
        # For 2 objectives, weights are (w, 1-w) for w in [0,1].
        # 16 values of w: 0, 1/15, 2/15, ..., 15/15
        
        weights = np.linspace(0, 1, n_cells)
        
        # For each cell, assign a weight and run Nelder-Mead
        # We need to be careful about the number of evaluations.
        # Nelder-Mead with maxiter and maxfun options.
        
        # Let's allocate: 16 cells, each gets 12 evaluations.
        # Nelder-Mead: initial simplex has dim+1 = 5 points. Then each iteration uses 1 evaluation.
        # So 5 + 7 = 12 evaluations per cell.
        
        # But we need to track evaluations globally.
        # Let's create a wrapper that counts evaluations and stops when budget is exceeded.
        
        eval_count = [0]
        
        def count_eval(x):
            if eval_count[0] >= budget:
                raise StopIteration
            eval_count[0] += 1
            return eval_point(x)
        
        # For each cell and weight, run Nelder-Mead
        for idx, (lo_c, hi_c) in enumerate(cell_bounds):
            if eval_count[0] >= budget:
                break
            
            w = weights[idx]
            # Scalarized objective: w * f0 + (1-w) * f1
            def scalar_obj(x):
                f = count_eval(x)
                return w * f[0] + (1 - w) * f[1]
            
            # Initial point: center of the cell
            x0 = (lo_c + hi_c) / 2.0
            
            # Run Nelder-Mead with tight constraints
            # maxiter: number of iterations, maxfun: max function evaluations
            # We want ~12 evaluations total. Initial simplex uses 5, so 7 more iterations.
            try:
                result = minimize(
                    scalar_obj,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 20,
                        'maxfev': 12,
                        'xatol': 1e-4,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
            except (StopIteration, Exception):
                pass
            
            # Also evaluate the best point found
            # Nelder-Mead already evaluated it, but let's make sure the result is in all_X/all_F
            # The count_eval wrapper already recorded all evaluations.
        
        # If we have remaining budget, do a second pass with different weight distributions
        if eval_count[0] < budget:
            # Use a finer set of weights for the remaining budget
            remaining = budget - eval_count[0]
            # Use 8 more evaluations: 8 cells with 1 evaluation each (just center)
            # Or 4 cells with 2 evaluations each
            # Let's just evaluate centers of cells with unused weights
            # Actually, let's do a few more scalarizations with intermediate weights
            extra_weights = np.linspace(0.1, 0.9, min(8, remaining))
            for w in extra_weights:
                if eval_count[0] >= budget:
                    break
                # Pick a random cell or the center of the whole space
                x0 = (lo + hi) / 2.0 + 0.1 * span * np.random.randn(dim)
                x0 = np.clip(x0, lo, hi)
                try:
                    def scalar_obj(x):
                        f = count_eval(x)
                        return w * f[0] + (1 - w) * f[1]
                    
                    result = minimize(
                        scalar_obj,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 10,
                            'maxfev': max(1, (budget - eval_count[0]) // max(1, len(extra_weights) - int(np.where(extra_weights == w)[0][0] if len(extra_weights) > 0 else 0))),
                            'xatol': 1e-4,
                            'fatol': 1e-6,
                            'adaptive': True
                        }
                    )
                except (StopIteration, Exception):
                    pass
        
        return all_X
# EVOLVE-BLOCK-END
