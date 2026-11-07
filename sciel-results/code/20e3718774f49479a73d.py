"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE21 with adaptive mutation and origin bias."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for the internal RNG to ensure reproducibility
        rng = np.random.default_rng(42)

        # Initial population: mix of origin-biased and uniform random
        pop_size = 30
        X = np.zeros((pop_size, dim))
        n_origin = pop_size // 3
        # Points near origin with small variance
        X[:n_origin] = 0.03 * span * rng.standard_normal((n_origin, dim))
        # Points with medium spread
        X[n_origin:2*n_origin] = 0.15 * span * rng.standard_normal((n_origin, dim))
        # Points uniformly random
        X[2*n_origin:] = lo + span * rng.random((pop_size - 2*n_origin, dim))
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Reference point for hypervolume calculation
        ref_point = np.array([3000.0, 0.06])

        def compute_hv(F, ref):
            """Compute hypervolume for 2 objectives with given reference point."""
            # Filter points that are strictly better than reference in all objectives
            valid = np.all(F < ref, axis=1)
            F_valid = F[valid]
            if len(F_valid) == 0:
                return 0.0
            
            # Sort by first objective (ascending)
            idx = np.argsort(F_valid[:, 0])
            F_sorted = F_valid[idx]
            
            hv = 0.0
            max_f2 = ref[1]  # Start from reference point's second objective
            
            for i in range(len(F_sorted)):
                f1, f2 = F_sorted[i]
                if f2 < max_f2:
                    hv += (max_f2 - f2) * (ref[0] - f1) if i == 0 else (max_f2 - f2) * (F_sorted[i-1, 0] - f1)
                    max_f2 = f2
            
            # Alternative correct computation
            # Sort by f1 ascending
            F_sorted = F_valid[np.argsort(F_valid[:, 0])]
            hv = 0.0
            prev_f2 = ref[1]
            for i in range(len(F_sorted)):
                f1, f2 = F_sorted[i]
                if f2 < prev_f2:
                    hv += (prev_f2 - f2) * (ref[0] - f1)
                    prev_f2 = f2
                # If f2 >= prev_f2, this point is dominated in the 2D HV sense
                # but we need to be careful. Let me use the standard algorithm.
            
            # Standard 2D HV algorithm:
            # Sort points by f1 ascending
            # Keep track of the minimum f2 seen so far (from the right)
            # Actually, for 2D: sort by f1 ascending, then for each point,
            # the contribution is (f2_prev - f2_curr) * (f1_curr - f1_prev) where
            # f1_prev is the previous point's f1 (or 0 if first) and f2_prev is
            # the previous point's f2 (or ref[1] if first)
            
            F_sorted = F_valid[np.argsort(F_valid[:, 0])]
            hv = 0.0
            f1_prev = 0.0
            f2_prev = ref[1]
            for i in range(len(F_sorted)):
                f1_curr, f2_curr = F_sorted[i]
                if f2_curr < f2_prev:
                    hv += (f2_prev - f2_curr) * (f1_curr - f1_prev)
                    f2_prev = f2_curr
                f1_prev = f1_curr
            
            return hv

        def is_dominated(F, i):
            """Check if point i is dominated by any other point."""
            for j in range(len(F)):
                if j == i:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    return True
            return False

        def get_non_dominated(F):
            """Get indices of non-dominated points."""
            nd = []
            for i in range(len(F)):
                if not is_dominated(F, i):
                    nd.append(i)
            return np.array(nd)

        def crowding_distance(F, nd_indices):
            """Compute crowding distance for non-dominated points."""
            if len(nd_indices) <= 2:
                return np.full(len(nd_indices), np.inf)
            
            cd = np.zeros(len(nd_indices))
            for obj in range(self.n_obj):
                f_vals = F[nd_indices, obj]
                sorted_idx = np.argsort(f_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = f_vals[sorted_idx[-1]] - f_vals[sorted_idx[0]]
                if f_range == 0:
                    continue
                for k in range(1, len(sorted_idx) - 1):
                    cd[sorted_idx[k]] += (f_vals[sorted_idx[k+1]] - f_vals[sorted_idx[k-1]]) / f_range
            return cd

        best_hv = compute_hv(F, ref_point)
        best_F = F.copy()

        # Adaptive mutation parameters
        base_scale = 0.25
        min_scale = 0.01

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Calculate progress for adaptive scaling
            progress = used / budget
            current_scale = base_scale * (1.0 - progress) + min_scale

            # Find non-dominated points
            nd_indices = get_non_dominated(F)
            
            if len(nd_indices) == 0:
                # Fallback to random
                parent_idx = rng.integers(pop_size)
            else:
                # Use crowding distance to select parent
                cd = crowding_distance(F, nd_indices)
                # Select parent with probability proportional to crowding distance
                cd_normalized = cd / (cd.sum() + 1e-10)
                parent_idx = nd_indices[rng.choice(len(nd_indices), p=cd_normalized)]
            
            parent = X[parent_idx]

            # Mutate with adaptive scale
            mutation = current_scale * span * rng.standard_normal(dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            used += 1

            # Determine replacement strategy:
            # If child is non-dominated, replace the point with lowest crowding distance
            # Otherwise, replace the worst point
            new_F = F.copy()
            new_F[parent_idx] = fc  # Temporarily add child
            
            # Check if child is dominated
            child_dominated = False
            for j in range(pop_size):
                if j == parent_idx:
                    continue
                if np.all(F[j] <= fc) and np.any(F[j] < fc):
                    child_dominated = True
                    break
            
            if not child_dominated:
                # Replace the point with lowest crowding distance among non-dominated
                nd = get_non_dominated(F)
                if len(nd) > 0:
                    cd_vals = crowding_distance(F, nd)
                    worst_idx = nd[np.argmin(cd_vals)]
                else:
                    worst_idx = parent_idx
            else:
                # Replace the worst point by sum of normalized objectives
                # Normalize objectives
                f_min = F.min(axis=0)
                f_max = F.max(axis=0)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0
                F_norm = (F - f_min) / f_range
                worst_idx = int(np.argmax(np.sum(F_norm, axis=1)))
            
            X[worst_idx] = child
            F[worst_idx] = fc

            # Update best HV if improved
            current_hv = compute_hv(F, ref_point)
            if current_hv > best_hv:
                best_hv = current_hv
                best_F = F.copy()

        return X
# EVOLVE-BLOCK-END
