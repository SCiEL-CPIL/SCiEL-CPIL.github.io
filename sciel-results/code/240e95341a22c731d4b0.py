"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Archive-based multi-objective optimizer with global/local search mix."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Initialize with a small random population
        init_size = min(10, self.budget // 2)
        X_init = lo + range_ * rng.random((init_size, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        
        # Archive: list of (x, f) tuples that are non-dominated
        archive_x = []
        archive_f = []
        
        def is_dominated(f_new, f_archive):
            # Returns True if f_new is dominated by any point in f_archive
            if len(f_archive) == 0:
                return False
            f_arr = np.array(f_archive)
            # f_new is dominated if exists f_a such that f_a <= f_new and f_a < f_new
            # Vectorized check:
            # For each f_a, check if all(f_a <= f_new) and any(f_a < f_new)
            # This is O(N_archive * n_obj). N_archive is small.
            for f_a in f_arr:
                if np.all(f_a <= f_new) and np.any(f_a < f_new):
                    return True
            return False

        def dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        # Add initial points to archive
        for i in range(init_size):
            x = X_init[i]
            f = F_init[i]
            if not is_dominated(f, archive_f):
                # Remove dominated points from archive
                new_archive_x = []
                new_archive_f = []
                for j in range(len(archive_f)):
                    if not dominates(f, archive_f[j]):
                        new_archive_x.append(archive_x[j])
                        new_archive_f.append(archive_f[j])
                new_archive_x.append(x)
                new_archive_f.append(f)
                archive_x = new_archive_x
                archive_f = new_archive_f
        
        used = init_size
        
        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Strategy: 50% global random, 50% local mutation of archive point
            if len(archive_x) > 0 and rng.random() < 0.5:
                # Local mutation
                idx = rng.integers(len(archive_x))
                parent = archive_x[idx]
                # Adaptive step size: start large, maybe shrink? 
                # For now, fixed small step relative to range
                step_size = 0.1 
                mutation = rng.standard_normal(self.dim) * step_size * range_
                child = parent + mutation
            else:
                # Global random
                child = lo + range_ * rng.random(self.dim)
            
            child = np.clip(child, lo, hi)
            f_child = problem(child)
            used += 1
            
            # Update archive
            if not is_dominated(f_child, archive_f):
                new_archive_x = []
                new_archive_f = []
                for j in range(len(archive_f)):
                    if not dominates(f_child, archive_f[j]):
                        new_archive_x.append(archive_x[j])
                        new_archive_f.append(archive_f[j])
                new_archive_x.append(child)
                new_archive_f.append(f_child)
                archive_x = new_archive_x
                archive_f = new_archive_f

        # Optional: If we have time, do a few local refinements on the best points?
        # With 200 budget, we likely used it all.
        
        return X_init # Return something, it's ignored
# EVOLVE-BLOCK-END
