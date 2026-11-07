# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using batched mutation with adaptive step sizes."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        rng = np.random.default_rng(42)

        # --- Phase 1: Larger initial sampling with LHS ---
        init_n = 30
        lhs = np.empty((init_n, dim))
        for d in range(dim):
            perm = rng.permutation(init_n)
            lhs[:, d] = (perm + rng.random(init_n)) / init_n
        X_init = lo + (hi - lo) * lhs
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = init_n

        archive_X = X_init.copy()
        archive_F = F_init.copy()

        # --- Helper: check if new point is dominated ---
        def is_dominated(point, archive_F):
            return np.any(np.all(archive_F <= point, axis=1) & np.any(archive_F < point, axis=1))

        # --- Helper: remove dominated points from archive ---
        def prune_archive(archive_X, archive_F):
            n = len(archive_F)
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                if not keep[i]:
                    continue
                for j in range(n):
                    if i == j or not keep[j]:
                        continue
                    if np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                        keep[i] = False
                        break
            return archive_X[keep], archive_F[keep]

        # --- Phase 2: Batched adaptive mutation ---
        remaining = budget - used
        if remaining > 0:
            # Adaptive step sizes: start larger, shrink over time
            step_start = 0.30
            step_end = 0.05
            total_iters = remaining
            
            for i in range(total_iters):
                if time.time() > deadline - 3:
                    break
                if used >= budget:
                    break
                
                # Linear interpolation of step size
                progress = i / max(1, total_iters - 1)
                step = step_start + (step_end - step_start) * progress
                
                # Generate a batch of children from different parents
                batch_size = min(10, budget - used)
                
                # Select parents: mix of non-dominated and random
                parents_X = []
                for _ in range(batch_size):
                    r = rng.random()
                    if r < 0.6 and len(archive_X) > 0:
                        # Pick from archive (bias toward non-dominated)
                        idx = rng.integers(len(archive_X))
                        parents_X.append(archive_X[idx])
                    else:
                        # Random point in bounds
                        parents_X.append(rng.uniform(lo, hi))
                
                # Generate children
                for parent in parents_X:
                    if used >= budget:
                        break
                    if time.time() > deadline - 3:
                        break
                    
                    # Gaussian mutation
                    child = parent + step * (hi - lo) * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    
                    # Check if dominated by archive
                    if not is_dominated(fc, archive_F):
                        # Add to archive and prune
                        archive_X = np.vstack([archive_X, child.reshape(1, -1)])
                        archive_F = np.vstack([archive_F, fc.reshape(1, -1)])
                        archive_X, archive_F = prune_archive(archive_X, archive_F)
                
                # Occasionally inject a fully random point for diversity
                if i % 20 == 0 and used < budget:
                    random_x = rng.uniform(lo, hi)
                    fc = problem(random_x)
                    used += 1
                    if not is_dominated(fc, archive_F):
                        archive_X = np.vstack([archive_X, random_x.reshape(1, -1)])
                        archive_F = np.vstack([archive_F, fc.reshape(1, -1)])
                        archive_X, archive_F = prune_archive(archive_X, archive_F)

        return archive_X

# EVOLVE-BLOCK-END
