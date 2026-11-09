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
        deadline = start_time + 285.0
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
        # ZDT4 structure: f1 = x1, f2 = g(x2..xn) * (1 - sqrt(x1/g(x2..xn)))
        # The Pareto front is formed when g(x2..xn) is minimized (g_min = 1).
        # For standard ZDT4 bounds, the global min of g is at x_i = 0.
        # However, we must verify this for the given bounds.
        
        # We allocate a small portion of the budget to find the best x_sec (x2..xn)
        # that minimizes g. We do this by minimizing f2 with x1 fixed at a small value
        # (or lo[0]) to isolate g.
        
        # We use the remaining budget to sweep x1 densely along the front.
        
        # Phase 1: Find a good x_sec (x2..xn) that minimizes g.
        # We try a few candidates:
        # 1. All zeros (global min for standard ZDT4)
        # 2. All pi (local min, often used in literature)
        # 3. Random samples to check for other minima
        
        # We will use a small fraction of the budget to find the best x_sec.
        n_phase1 = max(10, int(budget * 0.05))
        
        best_x_sec = None
        best_g_val = np.inf
        
        # Candidate 1: Zeros
        x_sec_zero = np.zeros(dim - 1)
        x_sec_zero = np.clip(x_sec_zero, lo[1:], hi[1:])
        
        # Candidate 2: Pi (clipped)
        x_sec_pi = np.full(dim - 1, np.pi)
        x_sec_pi = np.clip(x_sec_pi, lo[1:], hi[1:])
        
        # Candidate 3: Random
        x_sec_rand = lo[1:] + span[1:] * np.random.random(dim - 1)
        
        candidates = [x_sec_zero, x_sec_pi, x_sec_rand]
        
        for x_sec in candidates:
            if used >= n_phase1:
                break
            if is_deadline_exceeded():
                break
            # Use x1 = lo[0] to evaluate f2, which is proportional to g
            # f2 = g * (1 - sqrt(lo[0]/g))
            # If lo[0] is 0, f2 = g. If lo[0] > 0, f2 is still monotonic with g.
            x_full = np.concatenate(([lo[0]], x_sec))
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            used += 1
            # f[1] is f2. We want to minimize f2.
            if f[1] < best_g_val:
                best_g_val = f[1]
                best_x_sec = x_sec.copy()
        
        if best_x_sec is None:
            best_x_sec = x_sec_zero.copy()
            x_full = np.concatenate(([lo[0]], best_x_sec))
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            used += 1
            best_g_val = f[1]

        # Phase 2: Refine x_sec using local search to ensure we are at a local minimum of g.
        # This is crucial because if we are at a local minimum, the resulting front is better.
        # We use a simple coordinate descent on x2..xn to minimize f2 (with x1 fixed).
        max_refine_evals = max(5, int(budget * 0.05))
        current_x_sec = best_x_sec.copy()
        current_f2 = best_g_val
        step_size = 0.1 * span[1:].max()
        
        for _ in range(max_refine_evals):
            if is_deadline_exceeded():
                break
            if used >= n_phase1 + max_refine_evals:
                break
            
            improved = False
            for j in range(dim - 1):
                for direction in [1, -1]:
                    new_x_sec = current_x_sec.copy()
                    new_x_sec[j] += direction * step_size
                    new_x_sec = np.clip(new_x_sec, lo[1:], hi[1:])
                    
                    x_full = np.concatenate(([lo[0]], new_x_sec))
                    x_full = np.clip(x_full, lo, hi)
                    f = problem(x_full)
                    used += 1
                    
                    if f[1] < current_f2:
                        current_f2 = f[1]
                        current_x_sec = new_x_sec
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        best_x_sec = current_x_sec

        # Phase 3: Dense sweep of x1.
        # The Pareto front is defined by x1 in [lo[0], hi[0]] and x2..xn = best_x_sec.
        # We want to maximize hypervolume. A dense, evenly spaced set of points is optimal.
        
        remaining_budget = budget - used
        if remaining_budget < 10:
            n_sweep = max(1, remaining_budget)
        else:
            n_sweep = remaining_budget
            
        # Generate x1 values
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
        while used < budget and not is_deadline_exceeded():
            x_rand = lo + span * np.random.random(dim)
            f = problem(x_rand)
            used += 1
            update_archive(f, x_rand)

        return archive_X
# EVOLVE-BLOCK-END
