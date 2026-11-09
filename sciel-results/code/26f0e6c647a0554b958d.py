"""Multi-objective optimizer for WFG9: structured sweep + local refinement."""
import numpy as np
import time


class Optimizer:
    """Structured sweep along primary decision variables + Gaussian refinement."""

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

        # ---- Phase 1: Structured deterministic sweep ----
        # WFG9 has 12 variables. The first few variables often control the
        # main trade-off structure. We do a grid sweep on the first 3-4 vars
        # and randomize the rest.
        
        # Use ~60% of budget for structured sweep
        sweep_budget = int(budget * 0.6)
        
        # Sweep dimensions: pick 3 primary dimensions
        # In WFG9, typically x[0], x[1], x[2] are the "position" variables
        sweep_dims = [0, 1, 2]
        n_sweep_points = 5  # 5 points per sweep dim = 125 total
        remaining_dims = [d for d in range(dim) if d not in sweep_dims]
        
        rng = np.random.default_rng(42)
        
        # Generate grid points for sweep
        sweep_points = []
        for i in range(n_sweep_points):
            for j in range(n_sweep_points):
                for k in range(n_sweep_points):
                    x = np.zeros(dim)
                    # Set sweep dimensions
                    x[sweep_dims[0]] = lo[sweep_dims[0]] + span[sweep_dims[0]] * (i / (n_sweep_points - 1))
                    x[sweep_dims[1]] = lo[sweep_dims[1]] + span[sweep_dims[1]] * (j / (n_sweep_points - 1))
                    x[sweep_dims[2]] = lo[sweep_dims[2]] + span[sweep_dims[2]] * (k / (n_sweep_points - 1))
                    # Randomize remaining dimensions
                    for d in remaining_dims:
                        x[d] = lo[d] + span[d] * rng.random()
                    sweep_points.append(x)
        
        # Evaluate sweep points
        for x in sweep_points:
            if budget_used >= budget or now() > deadline:
                break
            eval_point(x)
        
        # ---- Phase 2: Local refinement using remaining budget ----
        # Find non-dominated points from what we have so far, then refine them
        
        def is_dominated(f, F_list):
            """Check if f is dominated by any point in F_list."""
            for fj in F_list:
                if np.all(fj <= f) and np.any(fj < f):
                    return True
            return False
        
        def get_non_dominated(X_list, F_list):
            """Return indices of non-dominated points."""
            n = len(F_list)
            non_dom = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_list[j] <= F_list[i]) and np.any(F_list[j] < F_list[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom.append(i)
            return non_dom
        
        # Use remaining budget for local refinement
        remaining_budget = budget - budget_used
        
        if remaining_budget > 0 and now() < deadline:
            # Get current non-dominated front
            if all_F:
                F_arr = np.array(all_F)
                X_arr = np.array(all_X)
                non_dom_idx = get_non_dominated(X_arr, F_arr)
                
                # Refine each non-dominated point with small perturbations
                # Allocate refinement budget evenly
                n_refine = len(non_dom_idx)
                if n_refine > 0:
                    budget_per_point = max(1, remaining_budget // n_refine)
                    
                    for idx in non_dom_idx:
                        if budget_used >= budget or now() > deadline:
                            break
                        x_base = X_arr[idx]
                        f_base = F_arr[idx]
                        
                        # Try multiple small perturbations
                        for trial in range(budget_per_point):
                            if budget_used >= budget or now() > deadline:
                                break
                            # Adaptive step size: start larger, get smaller
                            progress = trial / max(1, budget_per_point)
                            step = span * 0.1 * (1.0 - 0.7 * progress)
                            x_new = x_base + rng.standard_normal(dim) * step
                            x_new = np.clip(x_new, lo, hi)
                            
                            f_new = problem(x_new)
                            budget_used += 1
                            all_X.append(x_new)
                            all_F.append(f_new)
                            
                            # If it improves (dominates base or is non-dominated), update base
                            if np.all(f_new <= f_base) and np.any(f_new < f_base):
                                x_base = x_new
                                f_base = f_new
                            elif not is_dominated(f_new, [f_base]):
                                # Keep both; continue refining from new point
                                x_base = x_new
                                f_base = f_new
        
        # Return all evaluated points
        if all_X:
            return np.array(all_X)
        else:
            return np.empty((0, dim))
