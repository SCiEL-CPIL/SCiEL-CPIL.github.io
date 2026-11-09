"""Multi-objective optimizer for DTLZ7 exploiting known Pareto front structure with smooth local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7.
    
    Strategy:
    1. Sample points on the analytical Pareto front of DTLZ7 (bias toward high-hypervolume regions).
    2. For each front point, use scipy L-BFGS-B to optimize distance variables x[3:] to drive g→0.
    3. Final polish pass with tighter tolerances on the best archive points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        n_dist = n - 3  # number of distance variables

        budget_used = 0
        archive_x = []
        archive_f = []

        def dominates(a, b):
            """Check if a dominates b"""
            return np.all(a <= b) and np.any(a < b)

        def is_non_dominated(point_f, archive_f_list):
            """Check if point_f is non-dominated with respect to archive_f_list"""
            for af in archive_f_list:
                if dominates(af, point_f):
                    return False
            return True

        def add_to_archive(x, f):
            """Add point to archive if non-dominated"""
            if not is_non_dominated(f, archive_f):
                return
            # Remove dominated points
            new_x = []
            new_f = []
            for ax, af in zip(archive_x, archive_f):
                if not dominates(f, af):
                    new_x.append(ax)
                    new_f.append(af)
            archive_x[:] = new_x
            archive_f[:] = new_f
            archive_x.append(x)
            archive_f.append(f)

        def eval_and_count(x):
            """Evaluate x and increment budget counter. Returns f or None if budget exceeded."""
            nonlocal budget_used
            if budget_used >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            budget_used += 1
            return f

        # Phase 1: Generate front shape samples biased toward high-hypervolume regions
        # For DTLZ7, the Pareto front is parameterized by x[0], x[1], x[2]
        # Hypervolume contribution is highest where objectives are small (near origin)
        # x[0] and x[1] should be small, x[2] can vary
        
        # Use stratified sampling with bias toward low x[0], x[1]
        n_front = 25  # number of front points to optimize
        
        # Create biased samples: mix of uniform and concentrated near origin
        front_params = []
        
        # Corner points for full extent coverage
        corner_configs = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.5, 0.0),
            (0.5, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.0, 0.0, 0.5),
            (0.0, 0.25, 0.25),
            (0.25, 0.0, 0.25),
        ]
        for c in corner_configs:
            front_params.append(c)
        
        # Add stratified samples biased toward low values
        n_strat = n_front - len(front_params)
        for i in range(n_strat):
            # Use power-law distribution to bias toward low values
            x0 = np.random.beta(2, 5)  # biased toward 0
            x1 = np.random.beta(2, 5)  # biased toward 0
            x2 = np.random.uniform(0, 1)
            front_params.append((x0, x1, x2))
        
        # Phase 2: For each front point, optimize distance variables using L-BFGS-B
        # The objective for distance optimization: minimize sum of objectives (proxy for g)
        # Since we can't access g directly, we minimize the sum of the 3 objectives
        
        optimized_points = []
        
        for (x0, x1, x2) in front_params:
            if budget_used >= self.budget or time.time() > deadline:
                break
            
            # Create initial point with x[3:] = 0 (good starting point for DTLZ7)
            x_init = np.zeros(n)
            x_init[0] = x0
            x_init[1] = x1
            x_init[2] = x2
            # x_init[3:] = 0
            
            # Define objective for distance optimization
            # We minimize the sum of objectives as a proxy for minimizing g
            def dist_obj(x_full):
                # Clip to bounds
                x_clipped = np.clip(x_full, lo, hi)
                f = problem(x_clipped)
                budget_used += 1
                # Use sum of objectives as scalar objective
                # This works because when g=0, the objectives are minimized
                return np.sum(f)
            
            # Use L-BFGS-B to optimize only the distance variables
            # We'll wrap it to fix x[0:3] and optimize x[3:]
            def dist_obj_wrapped(x_dist):
                x_full = np.zeros(n)
                x_full[0] = x0
                x_full[1] = x1
                x_full[2] = x2
                x_full[3:] = x_dist
                x_full = np.clip(x_full, lo, hi)
                f = problem(x_full)
                budget_used += 1
                return np.sum(f)
            
            # Bounds for distance variables
            dist_lo = lo[3:]
            dist_hi = hi[3:]
            
            # Initial guess: zeros (known good for DTLZ7)
            x0_dist = np.zeros(n_dist)
            
            # Run L-BFGS-B with limited iterations to save budget
            try:
                result = minimize(
                    dist_obj_wrapped,
                    x0_dist,
                    method='L-BFGS-B',
                    bounds=list(zip(dist_lo, dist_hi)),
                    options={'maxiter': 10, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if result.success or result.fun < 1e-3:
                    x_opt = np.zeros(n)
                    x_opt[0] = x0
                    x_opt[1] = x1
                    x_opt[2] = x2
                    x_opt[3:] = result.x
                    x_opt = np.clip(x_opt, lo, hi)
                    f_opt = problem(x_opt)
                    budget_used += 1
                    optimized_points.append((x_opt.copy(), f_opt))
            except Exception:
                pass
        
        # Add all optimized points to archive
        for x, f in optimized_points:
            add_to_archive(x, f)
        
        # Phase 3: If budget remains, do local refinement around best archive points
        if budget_used < self.budget and len(archive_x) > 0:
            # Sort archive by sum of objectives
            sums = [np.sum(f) for f in archive_f]
            sorted_idx = np.argsort(sums)
            
            # Refine the top points with small perturbations
            n_refine = min(10, len(archive_x))
            for i in range(n_refine):
                if budget_used >= self.budget or time.time() > deadline:
                    break
                
                idx = sorted_idx[i]
                base_x = archive_x[idx].copy()
                
                # Try several perturbations
                for trial in range(3):
                    if budget_used >= self.budget or time.time() > deadline:
                        break
                    
                    # Small Gaussian perturbation
                    step = 0.01 * (1.0 + 0.5 * trial)
                    perturbation = np.random.normal(0, step, size=n)
                    child_x = base_x + perturbation
                    child_x = np.clip(child_x, lo, hi)
                    
                    child_f = problem(child_x)
                    budget_used += 1
                    
                    # Add if non-dominated
                    if is_non_dominated(child_f, archive_f):
                        add_to_archive(child_x.copy(), child_f)
        
        # Phase 4: Final polish - try to improve the best points with very tight optimization
        if budget_used < self.budget and len(archive_x) > 0:
            sums = [np.sum(f) for f in archive_f]
            sorted_idx = np.argsort(sums)
            
            n_polish = min(5, len(archive_x))
            for i in range(n_polish):
                if budget_used >= self.budget or time.time() > deadline:
                    break
                
                idx = sorted_idx[i]
                base_x = archive_x[idx].copy()
                
                # Polish the distance variables with tighter tolerance
                x0, x1, x2 = base_x[0], base_x[1], base_x[2]
                
                def polish_obj(x_dist):
                    x_full = np.zeros(n)
                    x_full[0] = x0
                    x_full[1] = x1
                    x_full[2] = x2
                    x_full[3:] = x_dist
                    x_full = np.clip(x_full, lo, hi)
                    f = problem(x_full)
                    budget_used += 1
                    return np.sum(f)
                
                try:
                    result = minimize(
                        polish_obj,
                        base_x[3:].copy(),
                        method='L-BFGS-B',
                        bounds=list(zip(lo[3:], hi[3:])),
                        options={'maxiter': 15, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if result.fun < np.sum(archive_f[idx]):
                        x_polished = np.zeros(n)
                        x_polished[0] = x0
                        x_polished[1] = x1
                        x_polished[2] = x2
                        x_polished[3:] = result.x
                        x_polished = np.clip(x_polished, lo, hi)
                        f_polished = problem(x_polished)
                        budget_used += 1
                        add_to_archive(x_polished.copy(), f_polished)
                except Exception:
                    pass
        
        # Return the archive
        if len(archive_x) > 0:
            return np.array(archive_x)
        return np.array([np.random.uniform(lo, hi)])
# EVOLVE-BLOCK-END
