"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE21 with archive-based search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        # Initialize with a mix of origin-biased and random points
        pop_size = 20
        X = np.zeros((pop_size, dim))
        n_origin = pop_size // 2
        X[:n_origin] = 0.03 * span * rng.standard_normal((n_origin, dim))
        X[n_origin:] = lo + span * rng.random((pop_size - n_origin, dim))
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Archive: list of (x, f) pairs that are non-dominated
        archive_X = list(X)
        archive_F = list(F)

        def get_non_dominated_indices(F_arr):
            """Return indices of non-dominated points in F_arr (n, m)."""
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                # Check if any other point dominates i
                # Point j dominates i if F[j] <= F[i] and F[j] < F[i]
                mask_le = np.all(F_arr <= F_arr[i], axis=1)
                mask_lt = np.any(F_arr < F_arr[i], axis=1)
                mask_le[i] = False
                if np.any(mask_le & mask_lt):
                    dominated[i] = True
            return np.where(~dominated)[0]

        def crowding_distance(F_nd, indices):
            """Compute crowding distance for non-dominated points."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            m = F_nd.shape[1]
            for k in range(m):
                f_k = F_nd[indices, k]
                min_k = f_k.min()
                max_k = f_k.max()
                if max_k - min_k < 1e-15:
                    continue
                order = np.argsort(f_k)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (f_k[order[i + 1]] - f_k[order[i - 1]]) / (max_k - min_k)
            return cd

        def add_to_archive(x, f):
            """Add a new point to the archive if it is non-dominated."""
            non_dom_idx = get_non_dominated_indices(np.array(archive_F))
            F_nd = np.array([archive_F[i] for i in non_dom_idx])
            if len(F_nd) == 0:
                archive_X.append(x)
                archive_F.append(f)
                return
            # Check if new point is dominated by any archive point
            mask_le = np.all(F_nd <= f, axis=1)
            mask_lt = np.any(F_nd < f, axis=1)
            if np.any(mask_le & mask_lt):
                return  # New point is dominated
            # Remove archive points dominated by the new point
            mask_le2 = np.all(f <= F_nd, axis=1)
            mask_lt2 = np.any(f < F_nd, axis=1)
            to_remove = np.where(mask_le2 & mask_lt2)[0]
            for idx in sorted(to_remove, reverse=True):
                global_idx = non_dom_idx[idx]
                archive_X.pop(global_idx)
                archive_F.pop(global_idx)
            # Re-check non-dominated set after removals
            non_dom_idx = get_non_dominated_indices(np.array(archive_F))
            archive_X.append(x)
            archive_F.append(f)

        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget
            # Adaptive mutation: start broad, narrow down
            current_scale = 0.25 * (1.0 - progress) + 0.01

            # Select parents from archive (non-dominated points)
            if len(archive_F) > 0:
                non_dom_idx = get_non_dominated_indices(np.array(archive_F))
                if len(non_dom_idx) == 0:
                    non_dom_idx = np.arange(len(archive_F))
                # Use crowding distance for selection
                F_nd = np.array([archive_F[i] for i in non_dom_idx])
                cd = crowding_distance(F_nd, np.arange(len(non_dom_idx)))
                # Select parent with highest crowding distance (most diverse)
                parent_local = int(np.argmax(cd))
                parent_global = non_dom_idx[parent_local]
                parent = np.array(archive_X[parent_global])
            else:
                parent = X[int(rng.integers(pop_size))]

            # Mutate with adaptive scale
            mutation = current_scale * span * rng.standard_normal(dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Add to archive
            add_to_archive(child, fc)

            # Update population: keep best points for future reference
            # Replace the point with worst scalarized objective
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
