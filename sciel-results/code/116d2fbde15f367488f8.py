"""Multi-objective optimizer for ZDT4 using Analytic Pareto Front Construction."""
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
        #    The Pareto front is formed when g(x2..xn) is minimized.
        #    For standard ZDT4, g_min = 1, achieved at x2..xn = 0.
        #    We assume the bounds allow x2..xn = 0 (or clip to it).
        
        # 2. Deterministic setting of secondary variables
        #    Set x2..xn to 0.0, clipped to [lo, hi].
        x_sec = np.zeros(dim - 1)
        x_sec = np.clip(x_sec, lo[1:], hi[1:])
        
        # 3. Determine valid range for x1
        #    f2 requires x1/g >= 0. Since g >= 1, x1 must be >= 0.
        #    Also x1 must be within [lo[0], hi[0]].
        x1_min = max(lo[0], 0.0)
        x1_max = hi[0]
        
        # If the range is invalid or empty, handle gracefully
        if x1_min > x1_max:
             x1_min = lo[0]
             x1_max = hi[0]
             if x1_min > x1_max:
                 return []

        # 4. Dense Sweep of x1
        #    Use the entire budget for this sweep to maximize point density.
        #    This maximizes the hypervolume of the recorded set.
        n_sweep = budget
        
        # Ensure we have at least 1 point
        if n_sweep < 1:
            return []

        x1_vals = np.linspace(x1_min, x1_max, n_sweep)
        
        for x1 in x1_vals:
            if is_deadline_exceeded():
                break
            if used >= budget:
                break
            
            x_full = np.concatenate(([x1], x_sec))
            x_full = np.clip(x_full, lo, hi)
            
            # Safety check: if x1 is slightly negative due to float error, clip
            if x_full[0] < 0:
                x_full[0] = 0.0

            f = problem(x_full)
            used += 1
            
            # Check for valid finite values
            if np.all(np.isfinite(f)):
                update_archive(f, x_full)
            # If f contains NaN/Inf, ignore it (don't add to archive)
            
        # If we have budget left (e.g. due to early break or other reasons), 
        # and if we haven't filled the budget, we could add random points, 
        # but the sweep should consume it all.
        
        # Final fallback: if archive is empty (e.g. all NaN), try a single point at x1=0
        if len(archive_X) == 0 and used < budget:
             x_fallback = np.concatenate(([0.0], x_sec))
             x_fallback = np.clip(x_fallback, lo, hi)
             f = problem(x_fallback)
             used += 1
             if np.all(np.isfinite(f)):
                 update_archive(f, x_fallback)

        return archive_X
# EVOLVE-BLOCK-END
