"""Rewritten program for WFG4 multi-objective optimization using local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 using structure-exploiting local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Archive storage
        archive_X = []
        archive_F = []
        used = 0
        
        def update_archive(x, f):
            nonlocal used
            # Check if f is dominated by any point in archive
            if len(archive_F) > 0:
                # Vectorized check: is there any point in archive that dominates f?
                # Domination: a <= b and a < b
                af = np.array(archive_F)
                dominated = ((af <= f).all(axis=1) & (af < f).any(axis=1)).any()
                if dominated:
                    return False
            
            # Remove points from archive dominated by f
            if len(archive_F) > 0:
                af = np.array(archive_F)
                ax = np.array(archive_X)
                # f dominates a[i] if f <= a[i] and f < a[i]
                dom_mask = (f <= af).all(axis=1) & (f < af).any(axis=1)
                if dom_mask.any():
                    keep = ~dom_mask
                    archive_F = [archive_F[i] for i in range(len(archive_F)) if keep[i]]
                    archive_X = [archive_X[i] for i in range(len(archive_X)) if keep[i]]
            
            archive_F.append(f)
            archive_X.append(x)
            return True

        # 1. Initialization: Spread points across the search space
        # WFG4 structure: x[0] controls position on front.
        # We initialize x[0] uniformly to cover the front, and other vars randomly.
        init_count = min(40, self.budget // 2)
        if init_count < 10:
            init_count = 10
            
        for i in range(init_count):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = np.random.uniform(lo, hi)
            # Bias x[0] to be evenly spaced to ensure coverage of the Pareto front
            x[0] = lo[0] + (hi[0] - lo[0]) * (i + np.random.rand()) / init_count
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
            update_archive(x, f)
            
        # 2. Local Search Phase
        # Select parents from archive and perform local perturbations
        # This exploits the smoothness of WFG4 to converge to the front.
        
        # Adaptive step size
        step_size = 0.1
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            if len(archive_X) == 0:
                # Fallback to random if archive is empty (should not happen)
                x = np.random.uniform(lo, hi)
            else:
                # Pick a random parent from the archive
                parent_idx = np.random.randint(0, len(archive_X))
                parent = archive_X[parent_idx]
                
                # Local mutation: Gaussian noise
                # Reduce step size as we approach the end of the budget to fine-tune
                progress = used / self.budget
                current_step = step_size * (1.0 - 0.5 * progress)
                
                x = parent + np.random.randn(self.dim) * current_step * span
                x = np.clip(x, lo, hi)
                
            f = problem(x)
            used += 1
            update_archive(x, f)
            
            # If we have a large archive, we might want to prune it to keep it diverse
            # But for 200 budget, memory is not an issue.
            
        # Final cleanup: filter by reference point
        ref = np.array([3.0, 5.0, 7.0])
        if len(archive_F) > 0:
            mask = np.array([(f < ref).all() for f in archive_F])
            if mask.any():
                final_X = np.array([archive_X[i] for i in range(len(archive_X)) if mask[i]])
                final_F = np.array([archive_F[i] for i in range(len(archive_F)) if mask[i]])
                return final_X
            else:
                return np.empty((0, self.dim))
        else:
            return np.empty((0, self.dim))

# EVOLVE-BLOCK-END
