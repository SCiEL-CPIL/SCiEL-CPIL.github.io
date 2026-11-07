"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
        deadline = start_time + 315.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref = np.array([1.1, 1.3, 1.1])

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def get_nondominated(F):
            if len(F) == 0:
                return np.array([], dtype=int)
            # F is (n, n_obj)
            # F[i] dominates F[j] if F[i] <= F[j] and F[i] != F[j]
            # We want indices i such that no j exists where F[j] dominates F[i]
            
            # Vectorized dominance check:
            # D[i, j] is True if F[i] dominates F[j]
            # F[i] <= F[j] for all obj -> (F[:, None, :] <= F[None, :, :]).all(axis=2)
            # F[i] < F[j] for at least one obj -> (F[:, None, :] < F[None, :, :]).any(axis=2)
            
            less_eq = (F[:, None, :] <= F[None, :, :]).all(axis=2)
            less = (F[:, None, :] < F[None, :, :]).any(axis=2)
            
            D = less_eq & less
            
            # i is non-dominated if no j exists such that D[j, i] is True
            # i.e., row i of D has no True in column i? No.
            # D[j, i] is True if j dominates i.
            # We want i such that sum(D[:, i]) == 0.
            
            dominated = D.any(axis=0)
            return np.where(~dominated)[0]

        def update_archive(archive_X, archive_F, new_X, new_F):
            # Add new point to archive, remove dominated points
            # new_X: (1, dim), new_F: (1, n_obj)
            
            # Check if new point is dominated by any existing point
            # existing_F: (k, n_obj)
            # D[j, 0] is True if existing_F[j] dominates new_F[0]
            
            if len(archive_F) > 0:
                less_eq = (archive_F[:, None, :] <= new_F[None, :, :]).all(axis=2)
                less = (archive_F[:, None, :] < new_F[None, :, :]).any(axis=2)
                D = less_eq & less
                # D has shape (k, 1). D[j, 0] is True if j dominates new.
                if D.any():
                    return archive_X, archive_F
            
            # Check if new point dominates any existing points
            # D[i, 0] is True if new_F[0] dominates existing_F[i]
            # Note: D in previous block was existing dominating new.
            # Let's compute new dominating existing.
            
            if len(archive_F) > 0:
                less_eq = (new_F[None, :, :] <= archive_F[:, None, :]).all(axis=2)
                less = (new_F[None, :, :] < archive_F[:, None, :]).any(axis=2)
                D_new_dom = less_eq & less
                # D_new_dom has shape (1, k). D_new_dom[0, i] is True if new dominates i.
                to_remove = D_new_dom[0, :]
            else:
                to_remove = np.array([], dtype=bool)
            
            # Remove dominated points
            if np.any(to_remove):
                mask = ~to_remove
                archive_X = archive_X[mask]
                archive_F = archive_F[mask]
            
            # Add new point
            archive_X = np.vstack([archive_X, new_X])
            archive_F = np.vstack([archive_F, new_F])
            
            return archive_X, archive_F

        # Phase 1: Initial Population
        # Use Latin Hypercube Sampling for initial spread
        n_init = min(100, budget)
        X = np.zeros((n_init, dim))
        for d in range(dim):
            edges = np.linspace(lo[d], hi[d], n_init + 1)
            for i in range(n_init):
                X[i, d] = edges[i] + np.random.random() * (edges[i+1] - edges[i])
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = n_init

        # Initialize archive
        nd_idx = get_nondominated(F)
        archive_X = X[nd_idx].copy()
        archive_F = F[nd_idx].copy()

        # Phase 2: Differential Evolution
        # Parameters
        CR = 0.8
        F_factor = 0.5 + 0.5 * np.random.random() # Adaptive F
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # If archive is too small, restart or random sample?
            # For now, if archive is empty, random sample.
            if len(archive_X) == 0:
                x_rand = np.random.uniform(lo, hi)
                f_rand = problem(x_rand)
                used += 1
                archive_X, archive_F = update_archive(archive_X, archive_F, x_rand.reshape(1, -1), f_rand.reshape(1, -1))
                continue
            
            # Select parents from archive
            # If archive size < 3, use random selection with replacement
            if len(archive_X) >= 3:
                idxs = np.random.choice(len(archive_X), size=3, replace=False)
                p1 = archive_X[idxs[0]]
                p2 = archive_X[idxs[1]]
                p3 = archive_X[idxs[2]]
            else:
                idxs = np.random.choice(len(archive_X), size=3, replace=True)
                p1 = archive_X[idxs[0]]
                p2 = archive_X[idxs[1]]
                p3 = archive_X[idxs[2]]
            
            # Generate trial vector
            # v = p1 + F * (p2 - p3)
            v = p1 + F_factor * (p2 - p3)
            
            # Clip to bounds
            v = np.clip(v, lo, hi)
            
            # Evaluate
            f_v = problem(v)
            used += 1
            
            # Update archive
            archive_X, archive_F = update_archive(archive_X, archive_F, v.reshape(1, -1), f_v.reshape(1, -1))
            
            # Optional: If archive grows too large, prune it to keep top N by hypervolume contribution?
            # Or just keep all non-dominated. For 200 budget, archive size should be manageable.
            # However, if archive becomes huge, DE selection might be slow.
            # Let's cap archive size at 100. If larger, remove point with lowest HV contribution.
            if len(archive_X) > 100:
                # Calculate HV contribution for each point in archive
                # This is O(N^2 * n_obj) roughly.
                # Simplified: remove point that is closest to another point?
                # Or just keep the first 100? No, that's biased.
                # Let's use a simple distance-based pruning if it gets too big.
                # Actually, with budget 200, max archive size is 200.
                # 200 points in 3D is fine for O(N^2) dominance check? 200^2 = 40000 ops. Fast enough.
                pass

        return X

# EVOLVE-BLOCK-END
