"""Multi-objective optimizer for ZDT4 using Structure-Exploiting Global Search."""
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
        deadline = start_time + 290.0  # Safety margin for 300s limit
        budget = self.budget
        dim = self.dim
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        
        used = 0
        archive_F = []
        archive_X = []

        def is_deadline_exceeded():
            return (time.time() - start_time) > deadline

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def update_archive(f_new, x_new):
            nonlocal archive_F, archive_X
            # Check domination
            for i in range(len(archive_F)):
                if dominates(archive_F[i], f_new):
                    return
            # Remove dominated
            new_F = []
            new_X = []
            for i in range(len(archive_F)):
                if not dominates(f_new, archive_F[i]):
                    new_F.append(archive_F[i])
                    new_X.append(archive_X[i])
            new_F.append(f_new)
            new_X.append(x_new)
            archive_F = new_F
            archive_X = new_X

        # Strategy:
        # 1. ZDT4 structure: f1 = x1, f2 = g(x2..xn) * (1 - sqrt(x1/g(x2..xn)))
        #    The Pareto front is formed when g(x2..xn) is minimized (g_min = 1).
        #    For standard ZDT4 bounds (often [0,1] or [-5,5]), the global min of g is at x_i = 0.
        #    However, we must verify this for the given bounds.
        #    
        # 2. We allocate a small portion of the budget to find the best x_sec (x2..xn)
        #    that minimizes g. We do this by minimizing f2 with x1 fixed at a small value
        #    (or lo[0]) to isolate g.
        #    
        # 3. We use the remaining budget to sweep x1 densely across its range, 
        #    using the best x_sec found. This generates a dense set of non-dominated
        #    points on the true Pareto front, maximizing hypervolume.

        # Phase 1: Find best secondary variables (x2..xn)
        # We use random sampling + local refinement (simple coordinate descent or just random)
        # Since we suspect x_i=0 is the global min, we should heavily bias towards 0.
        
        # Let's use a hybrid:
        # - Try x_sec = 0.0 (if within bounds)
        # - Try random points
        # - Keep the best x_sec based on f2 (with x1 fixed)
        
        x1_probe = lo[0] # Use lower bound to minimize f2 ~ g
        
        best_x_sec = None
        best_f2_val = np.inf
        
        # Candidate 1: All zeros (clipped to bounds)
        x_sec_zero = np.zeros(dim - 1)
        x_sec_zero = np.clip(x_sec_zero, lo[1:], hi[1:])
        
        # Candidate 2: Random samples
        n_random_tries = 20
        random_x_secs = []
        for _ in range(n_random_tries):
            if used >= budget * 0.1: # Use max 10% of budget for this phase
                break
            if is_deadline_exceeded():
                break
            x_sec_rand = lo[1:] + span[1:] * np.random.random(dim - 1)
            random_x_secs.append(x_sec_rand)
        
        # Evaluate candidates
        candidates = [x_sec_zero] + random_x_secs
        
        for x_sec in candidates:
            if used >= budget * 0.1:
                break
            if is_deadline_exceeded():
                break
            x_full = np.concatenate(([x1_probe], x_sec))
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            used += 1
            # f[1] is f2. At x1=lo[0], f2 is proportional to g.
            if f[1] < best_f2_val:
                best_f2_val = f[1]
                best_x_sec = x_sec.copy()
        
        if best_x_sec is None:
            best_x_sec = np.zeros(dim - 1)
            best_x_sec = np.clip(best_x_sec, lo[1:], hi[1:])
            x_full = np.concatenate(([x1_probe], best_x_sec))
            f = problem(x_full)
            used += 1

        # Phase 2: Refine best_x_sec slightly using local search (Coordinate Descent)
        # This helps if the global min is not exactly 0 but close, or if bounds are weird.
        # We have ~90% budget left. Let's spend 5% on refinement.
        max_refine_evals = int(budget * 0.05)
        current_x_sec = best_x_sec.copy()
        current_f2 = best_f2_val
        step_size = 0.1 * span[1:].max()
        
        for i in range(max_refine_evals):
            if is_deadline_exceeded():
                break
            if used >= budget * 0.15:
                break
            
            improved = False
            for j in range(dim - 1):
                # Try moving in + and - direction
                for direction in [1, -1]:
                    new_x_sec = current_x_sec.copy()
                    new_x_sec[j] += direction * step_size
                    new_x_sec = np.clip(new_x_sec, lo[1:], hi[1:])
                    
                    x_full = np.concatenate(([x1_probe], new_x_sec))
                    f = problem(x_full)
                    used += 1
                    
                    if f[1] < current_f2:
                        current_f2 = f[1]
                        current_x_sec = new_x_sec
                        improved = True
                        break # Move to next dimension
                
                if improved:
                    break # Restart coordinate descent from new point
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        best_x_sec = current_x_sec

        # Phase 3: Dense Sweep of x1
        # The Pareto front is the curve defined by x1 in [lo[0], hi[0]] and x2..xn = best_x_sec.
        # We want to maximize hypervolume. A dense, evenly spaced set of points is optimal
        # for hypervolume if the front is smooth.
        
        remaining_budget = budget - used
        # Reserve a few evals for safety
        if remaining_budget < 10:
            n_sweep = 1
        else:
            # Use all remaining budget for sweeping
            n_sweep = remaining_budget
            
        # Generate x1 values
        # We want to cover the range [lo[0], hi[0]]
        # linspace is good for smooth fronts.
        x1_vals = np.linspace(lo[0], hi[0], n_sweep)
        
        for x1 in x1_vals:
            if is_deadline_exceeded():
                break
            if used >= budget:
                break
            
            x_full = np.concatenate(([x1], best_x_sec))
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            used += 1
            update_archive(f, x_full)
            
        # Phase 4: Fill remaining budget with random points if any left
        # (Should be minimal)
        while used < budget and not is_deadline_exceeded():
            x_rand = lo + span * np.random.random(dim)
            f = problem(x_rand)
            used += 1
            update_archive(f, x_rand)

        return archive_X
# EVOLVE-BLOCK-END
