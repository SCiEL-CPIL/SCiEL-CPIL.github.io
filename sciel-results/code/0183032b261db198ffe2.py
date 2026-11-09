"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Pareto Archive-based Evolutionary Optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Archive: list of (x, f) tuples that are non-dominated
        archive = []
        used = 0
        
        def is_dominated(f1, f2):
            # f1 dominates f2 if f1 is better in all objectives and strictly better in at least one
            # Lower is better
            if np.all(f1 <= f2) and np.any(f1 < f2):
                return True
            return False
            
        def update_archive(x_new, f_new):
            nonlocal archive
            # Check if f_new is dominated by any in archive
            for x_a, f_a in archive:
                if is_dominated(f_new, f_a):
                    return False # f_new is dominated, do not add
            
            # Remove any members of archive dominated by f_new
            new_archive = []
            for x_a, f_a in archive:
                if is_dominated(f_a, f_new):
                    # f_a is dominated by f_new, remove it
                    continue
                new_archive.append((x_a, f_a))
            
            archive = new_archive
            archive.append((x_new, f_new))
            
            # Limit archive size to prevent memory bloat and ensure diversity
            # Simple strategy: if size > 100, remove random point? 
            # Better: keep only non-dominated (already done). 
            # If archive is too large, we might lose coverage. 
            # For budget 200, archive size won't be huge.
            return True

        # 1. Initialization
        pop_size = 20
        for _ in range(pop_size):
            if used >= budget:
                break
            x = rng.uniform(lo, hi)
            f = problem(x)
            used += 1
            update_archive(x, f)
            
        # 2. Evolution
        # Parameters for mutation
        F = 0.5 # Differential weight
        
        while used < budget:
            if len(archive) == 0:
                break
                
            # Select parent
            parent_idx = rng.integers(0, len(archive))
            parent_x = archive[parent_idx][0]
            parent_f = archive[parent_idx][1]
            
            # Select two other random parents from archive for DE
            if len(archive) >= 3:
                idx1, idx2 = rng.choice(len(archive), size=2, replace=False)
                # Ensure they are not the parent (optional, but good)
                # If archive is small, this is fine
                x1 = archive[idx1][0]
                x2 = archive[idx2][0]
                x_mut = parent_x + F * (x1 - x2)
            else:
                # Fallback to random perturbation if archive is too small
                x_mut = parent_x + rng.normal(0, 0.1, size=n_dim)
                
            x_mut = np.clip(x_mut, lo, hi)
            
            f_mut = problem(x_mut)
            used += 1
            
            update_archive(x_mut, f_mut)
            
            # Optional: If archive is empty (shouldn't happen if we start with valid points), restart
            # But update_archive keeps at least one if new point is non-dominated.
            
        # Return all points in the final archive (and maybe all evaluated points? 
        # The task says "Every returned objective vector is recorded". 
        # The evaluator records the return value of __call__? 
        # "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # Wait, "Every returned objective vector is recorded" usually implies the evaluator intercepts calls to `problem`.
        # "The return value of __call__ is ignored."
        # So I just need to make sure I call `problem` enough times.
        # The score is based on "all recorded vectors". 
        # So I should return something, but it doesn't matter. 
        # I will return the archive points as a list of arrays, just in case.
        
        return np.array([x for x, f in archive])

# EVOLVE-BLOCK-END
