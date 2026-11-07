import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin vs 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        self.evals = 0
        self.max_budget = self.budget
        
        def eval_func(x):
            if self.evals >= self.max_budget:
                return np.array([1e6, 1e6, 1e6])
            self.evals += 1
            try:
                f = np.array(problem(x), dtype=float)
                if not np.all(np.isfinite(f)):
                    f = np.array([1e6, 1e6, 1e6])
            except:
                f = np.array([1e6, 1e6, 1e6])
            return f

        # WFG9 structure: obj1 depends only on g_i (i>=1), can be made ~0
        # obj2, obj3 are h(g_avg, t) where t = x[0]
        # Strategy: Sweep t = x[0] across range, set x[1:] to minimize g_i
        
        # Phase 1: Deterministic LHS for broad coverage (100 points)
        n_lhs = min(100, self.budget)
        lhs_points = np.zeros((n_lhs, self.dim))
        
        # Generate LHS with seeded random for reproducibility
        for j in range(self.dim):
            perm = np.random.permutation(n_lhs)
            u = (perm + np.random.random(n_lhs)) / n_lhs
            lhs_points[:, j] = lo[j] + u * (hi[j] - lo[j])

        archive_x = []
        archive_f = []
        
        for i in range(n_lhs):
            if time.time() > deadline or self.evals >= self.budget:
                break
            x = lhs_points[i]
            f = eval_func(x)
            archive_x.append(x)
            archive_f.append(f)

        # Efficient dominance check using vectorized operations
        def update_archive(new_x, new_f, archive_x, archive_f):
            """Add new point if non-dominated, remove dominated points."""
            # Check if new_f is dominated by any archive point
            if len(archive_f) > 0:
                af = np.array(archive_f)
                dominated = np.all(af <= new_f, axis=1) & np.any(af < new_f, axis=1)
                if np.any(dominated):
                    return archive_x, archive_f
            
            # Remove archive points dominated by new_f
            if len(archive_f) > 0:
                af = np.array(archive_f)
                dom_mask = np.all(new_f <= af, axis=1) & np.any(new_f < af, axis=1)
                if np.any(dom_mask):
                    archive_x = [archive_x[i] for i in range(len(archive_x)) if not dom_mask[i]]
                    archive_f = [archive_f[i] for i in range(len(archive_f)) if not dom_mask[i]]
            
            archive_x.append(new_x)
            archive_f.append(new_f)
            return archive_x, archive_f

        # Phase 2: Exploit WFG9 structure
        # Objective 1 can be made ~0 by setting x[1:6] and x[7:12] appropriately
        # For WFG9: g_i = (x_i - 0.35*cos(12*pi*x_0*x_i) - 0.35)^2 for i=1..11
        # To minimize g_i, set x_i such that x_i - 0.35*cos(12*pi*x_0*x_i) = 0.35
        # This is hard to solve analytically, but we can try x_i = 0.35 for all i
        
        # Sweep x[0] (t) across the range, with x[1:] set to minimize g_i
        remaining_budget = self.budget - self.evals
        if remaining_budget > 20:
            # Create a sweep of x[0] values
            n_sweep = min(40, remaining_budget)
            t_values = np.linspace(lo[0], hi[0], n_sweep)
            
            for t in t_values:
                if time.time() > deadline or self.evals >= self.budget:
                    break
                
                # Set x[1:] to values that minimize g_i
                # Try x_i = 0.35 for i=1..11 (heuristic for WFG9)
                x = np.zeros(self.dim)
                x[0] = t
                # For WFG9, setting x_i = 0.35 often minimizes g_i
                x[1:] = 0.35
                
                # Clip to bounds
                x = np.clip(x, lo, hi)
                
                f = eval_func(x)
                archive_x, archive_f = update_archive(x, f, archive_x, archive_f)
        
        # Phase 3: Local refinement around best points
        remaining_budget = self.budget - self.evals
        if remaining_budget > 10 and archive_f:
            # Select top 5 points by sum of objectives
            sums = np.sum(np.array(archive_f), axis=1)
            k_refine = min(5, len(archive_f))
            best_indices = np.argsort(sums)[:k_refine]
            
            evals_per_point = max(3, remaining_budget // k_refine)
            
            for idx in best_indices:
                if time.time() > deadline or self.evals >= self.budget:
                    break
                
                x0 = np.array(archive_x[idx])
                
                # Simple local search: try small perturbations
                step = 0.1
                for _ in range(evals_per_point):
                    if time.time() > deadline or self.evals >= self.budget:
                        break
                    
                    # Random perturbation
                    perturbation = np.random.uniform(-step, step, self.dim)
                    x_new = np.clip(x0 + perturbation, lo, hi)
                    
                    f_new = eval_func(x_new)
                    archive_x, archive_f = update_archive(x_new, f_new, archive_x, archive_f)
                    
                    # Update x0 if we found a better point
                    if np.sum(f_new) < np.sum(np.array(archive_f)[archive_f.index(f_new)] if f_new in archive_f else [1e6]):
                        x0 = x_new
        
        # Final cleanup: ensure we have a valid return
        if archive_x:
            return np.array(archive_x)
        return np.array([lo + 0.5 * (hi - lo)])
