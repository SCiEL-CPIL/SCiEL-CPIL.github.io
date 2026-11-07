# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin (330s limit, keep 50s buffer)
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        mid = (lo + hi) / 2.0
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # Track evaluations
        evals_used = 0
        
        # Archive: list of (x, f) tuples
        archive = []
        
        # Helper: check if f is dominated by any point in archive
        def is_dominated_by_archive(f_new):
            if not archive:
                return False
            for _, f_a in archive:
                if np.all(f_a <= f_new) and np.any(f_a < f_new):
                    return True
            return False
        
        # Helper: add to archive if non-dominated
        def add_to_archive(x, f):
            nonlocal evals_used
            # Check if f is dominated by any existing archive point
            if is_dominated_by_archive(f):
                return False
            # Remove archive points dominated by f
            new_archive = []
            for x_a, f_a in archive:
                if np.all(f <= f_a) and np.any(f < f_a):
                    continue
                new_archive.append((x_a, f_a))
            new_archive.append((x, f))
            archive.clear()
            archive.extend(new_archive)
            return True
        
        # Helper: evaluate and add to archive
        def evaluate(x):
            nonlocal evals_used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            evals_used += 1
            add_to_archive(x.copy(), f)
            return x, f
        
        # Check time
        def time_exceeded():
            return (time.time() - start_time) > (deadline - 5.0)
        
        # ====================================================================
        # PHASE 1: Chebyshev Scalarization with Systematic Weight Vectors
        # ====================================================================
        # For 3 objectives, we sample weight vectors on the simplex
        # w1 + w2 + w3 = 1, wi >= 0
        
        # Generate weight vectors: use a grid on the 2D simplex
        # For 3 obj, simplex is a triangle. We can parameterize by (w1, w2) with w3 = 1 - w1 - w2
        # w1 in [0, 1], w2 in [0, 1 - w1]
        
        # Number of scalarizations: use ~40% of budget
        n_scalar = max(10, budget // 3)
        
        # Generate diverse weight vectors
        # Use a stratified approach: for each "layer" of w1, vary w2
        weights = []
        # Add extreme weights
        weights.append([1.0, 0.0, 0.0])
        weights.append([0.0, 1.0, 0.0])
        weights.append([0.0, 0.0, 1.0])
        # Add balanced
        weights.append([1/3.0, 1/3.0, 1/3.0])
        
        # Generate more weights on the simplex
        n_extra = n_scalar - len(weights)
        for _ in range(n_extra):
            # Random point on simplex using Dirichlet(1,1,1)
            r = np.random.dirichlet([1.0, 1.0, 1.0])
            weights.append(r)
        
        weights = np.array(weights)
        
        # For each weight vector, solve the Chebyshev scalarization subproblem:
        # min_x max_i (w_i * f_i(x) - z_i*)
        # where z* is the ideal point (approximated)
        
        # First, get a rough estimate of the ideal point from initial random samples
        n_init = min(10, budget - n_scalar - 10)
        if n_init > 0:
            for _ in range(n_init):
                x = np.random.uniform(lo, hi)
                evaluate(x)
                if time_exceeded() or evals_used >= budget:
                    break
        
        # Estimate ideal point from archive
        if archive:
            ideal = np.array([min(f for _, f in archive)])
            # Actually, ideal point is component-wise minimum
            ideal = np.array([min(f[i] for _, f in archive) for i in range(n_obj)])
        else:
            ideal = np.zeros(n_obj)
        
        # Now solve each scalarization subproblem using local search (Nelder-Mead style)
        # For each weight, we use a few starting points and do local search
        
        evals_per_scalar = max(3, (budget - evals_used) // max(1, len(weights)))
        
        for w in weights:
            if time_exceeded() or evals_used >= budget:
                break
            
            # Chebyshev scalarization objective:
            # g(x) = max_i (w_i * f_i(x) - w_i * ideal_i)
            # But we need to handle the ideal point properly.
            # Actually, standard Chebyshev: min max_i (w_i * (f_i(x) - z_i^*))
            # where z^* is a reference point (ideal point)
            
            def cheb_obj(x):
                f = np.array(problem(x), dtype=np.float64)
                # Normalize by ideal point differences
                # Use the current ideal estimate
                diffs = f - ideal
                # Weighted
                weighted = w * diffs
                return np.max(weighted)
            
            # Use multiple starting points for robustness
            n_starts = min(3, evals_per_scalar // 2)
            
            for s in range(n_starts):
                if time_exceeded() or evals_used >= budget:
                    break
                
                # Starting point: mix of random and from archive
                if s == 0 and archive:
                    # Start from a random archive point
                    idx = np.random.randint(len(archive))
                    x0 = archive[idx][0].copy()
                elif s == 1:
                    x0 = mid.copy()
                else:
                    x0 = np.random.uniform(lo, hi)
                
                # Simple local search: random restarts with shrinking step
                x_cur = x0.copy()
                # Evaluate starting point
                x_cur, f_cur = evaluate(x_cur)
                
                # Local search: try perturbations
                step_size = 0.1 * span
                for step in range(max(2, evals_per_scalar // n_starts)):
                    if time_exceeded() or evals_used >= budget:
                        break
                    
                    # Try several random directions
                    best_x = x_cur
                    best_f = f_cur
                    best_cheb = np.max(w * (best_f - ideal))
                    
                    for _ in range(3):
                        x_try = x_cur + np.random.normal(0, step_size / 10.0, dim)
                        x_try = np.clip(x_try, lo, hi)
                        f_try = np.array(problem(x_try), dtype=np.float64)
                        evals_used += 1
                        add_to_archive(x_try.copy(), f_try)
                        
                        cheb_try = np.max(w * (f_try - ideal))
                        if cheb_try < best_cheb:
                            best_x = x_try
                            best_f = f_try
                            best_cheb = cheb_try
                    
                    if np.array_equal(best_x, x_cur):
                        # No improvement, shrink step
                        step_size *= 0.7
                    else:
                        x_cur = best_x
                        f_cur = best_f
        
        # ====================================================================
        # PHASE 2: Fill remaining budget with random sampling + local refinement
        # ====================================================================
        if evals_used < budget and not time_exceeded():
            # Use remaining budget for random sampling to find more non-dominated points
            while evals_used < budget and not time_exceeded():
                x = np.random.uniform(lo, hi)
                evaluate(x)
        
        # ====================================================================
        # PHASE 3: Local refinement of best archive points
        # ====================================================================
        if archive and evals_used < budget and not time_exceeded():
            # Sort archive by hypervolume contribution proxy (distance to reference)
            ref = np.array([3.0, 5.0, 7.0])
            # For each archive point, compute its "quality" as sum of (ref - f) for positive components
            archive_sorted = sorted(archive, key=lambda xf: np.sum(np.maximum(ref - xf[1], 0)))
            
            n_refine = min(len(archive_sorted), max(5, (budget - evals_used) // 10))
            
            for x_a, f_a in archive_sorted[:n_refine]:
                if time_exceeded() or evals_used >= budget:
                    break
                
                x_cur = x_a.copy()
                f_cur = f_a.copy()
                step_size = 0.05 * span
                
                for _ in range(max(2, (budget - evals_used) // max(1, n_refine))):
                    if time_exceeded() or evals_used >= budget:
                        break
                    
                    x_try = x_cur + np.random.normal(0, step_size / 10.0, dim)
                    x_try = np.clip(x_try, lo, hi)
                    f_try = np.array(problem(x_try), dtype=np.float64)
                    evals_used += 1
                    add_to_archive(x_try.copy(), f_try)
                    
                    # Accept if non-dominated
                    if not is_dominated_by_archive(f_try):
                        # Check if it dominates current
                        if np.all(f_try <= f_cur) and np.any(f_try < f_cur):
                            x_cur = x_try
                            f_cur = f_try
                    step_size *= 0.8
        
        return archive
# EVOLVE-BLOCK-END
