"""Multi-objective optimizer for WFG9: exploit analytic structure with dense grid + refinement."""
import numpy as np
import time


class Optimizer:
    """Dense grid sweep on WFG9's 2-D position subspace + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 295.0  # safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        def now():
            return time.time()

        # State for tracking all evaluated points
        all_X = []  # list of x vectors
        all_F = []  # list of F vectors

        budget_used = 0

        def eval_point(x):
            nonlocal budget_used
            if budget_used >= budget or now() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            budget_used += 1
            all_X.append(x)
            all_F.append(f)
            return f

        # WFG9 structure: objectives depend on x[0], x[1] (position) and x[2] (shape).
        # Fix x[2] to a value that minimizes the shape effect.
        # In standard WFG9, x[2] controls the "k" parameter. Setting it to mid-range
        # or a specific optimal value helps. We'll fix x[2] = 0.5 * (lo[2] + hi[2])
        # and x[3..11] to midpoints as well.
        
        # Determine which variables to sweep. For WFG9, typically x[0] and x[1] are position.
        # Let's assume the first 2 variables are the key position variables.
        sweep_dims = [0, 1]
        fixed_dims = [d for d in range(dim) if d not in sweep_dims]
        
        # Set fixed variables to their midpoints
        fixed_vals = {}
        for d in fixed_dims:
            fixed_vals[d] = 0.5 * (lo[d] + hi[d])
        
        # For WFG9, x[2] is often the shape variable. Let's try a few values for x[2]
        # to see if it helps, but for now, let's just use midpoint for all non-swept.
        
        # Dense grid over the 2-D subspace
        # Budget allocation: ~150 for grid, ~50 for refinement
        n_grid = 12  # 12x12 = 144 points
        grid_budget = n_grid * n_grid
        
        if grid_budget > budget:
            # Adjust grid size to fit budget
            n_grid = int(np.sqrt(budget * 0.75))
            grid_budget = n_grid * n_grid
        
        # Create grid points
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)
        
        # Evaluate grid
        for x0 in x0_vals:
            if budget_used >= budget or now() > deadline:
                break
            for x1 in x1_vals:
                if budget_used >= budget or now() > deadline:
                    break
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                for d in fixed_dims:
                    x[d] = fixed_vals[d]
                eval_point(x)
        
        # Local refinement around the best non-dominated points
        if all_F and budget_used < budget:
            F_arr = np.array(all_F)
            
            # Find non-dominated points in the grid
            def is_non_dominated(f, F_all):
                for j, fj in enumerate(F_all):
                    if j != f:
                        if np.all(fj <= f) and np.any(fj < f):
                            return False
                return True
            
            non_dom_indices = []
            for i, f in enumerate(F_arr):
                if is_non_dominated(f, F_arr):
                    non_dom_indices.append(i)
            
            # Refine around top non-dominated points
            n_refine = min(50, budget - budget_used)
            if n_refine > 0 and non_dom_indices:
                # Select a few diverse non-dominated points to refine
                # Use simple selection: take first few non-dominated
                refine_centers = non_dom_indices[:min(5, len(non_dom_indices))]
                
                for center_idx in refine_centers:
                    if budget_used >= budget or now() > deadline:
                        break
                    
                    center_x = all_X[center_idx].copy()
                    center_f = F_arr[center_idx]
                    
                    # Local Gaussian refinement
                    n_local = n_refine // max(1, len(refine_centers))
                    for _ in range(n_local):
                        if budget_used >= budget or now() > deadline:
                            break
                        
                        # Perturb only the sweep dimensions
                        x_new = center_x.copy()
                        for d in sweep_dims:
                            x_new[d] += np.random.normal(0, 0.1 * span[d])
                        
                        eval_point(x_new)
        
        # Return all evaluated points
        if all_X:
            return np.array(all_X)
        else:
            return np.empty((0, dim))
