"""Multi-objective optimizer for RE21 with budget 200, dim 4, n_obj 2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using hypervolume-based selection and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (used only for selection, not scoring)
        ref = np.array([3000.0, 0.06])

        def dominates(a, b):
            """True if a dominates b (a <= b in all objectives, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_non_dominated(F):
            """Return boolean mask of non-dominated points in F."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i != j and mask[j] and dominates(F[j], F[i]):
                        mask[i] = False
                        break
            return mask

        def hypervolume_2d(F, ref):
            """Compute hypervolume for 2D objective space. F is (n, 2) array.
            Only points strictly better than ref in every objective count."""
            # Filter points that are strictly better than ref
            valid = np.all(F < ref, axis=1)
            Fv = F[valid]
            if len(Fv) == 0:
                return 0.0
            # Sort by first objective (ascending)
            idx = np.argsort(Fv[:, 0])
            Fv = Fv[idx]
            # Keep only non-dominated points
            nd = is_non_dominated(Fv)
            Fv = Fv[nd]
            if len(Fv) == 0:
                return 0.0
            # Compute hypervolume using sweep line
            hv = 0.0
            prev_y = ref[1]
            for i in range(len(Fv)):
                x = Fv[i, 0]
                y = Fv[i, 1]
                if x < ref[0] and y < ref[1]:
                    width = min(ref[0], x + 1e-15) - Fv[i, 0] if i < len(Fv) - 1 else ref[0] - Fv[i, 0]
                    # Actually, let's do it properly: sort by x ascending
                    pass
            # Proper 2D hypervolume:
            # Sort by x ascending, then sweep
            Fv_sorted = Fv[np.argsort(Fv[:, 0])]
            hv = 0.0
            max_y = ref[1]
            for i in range(len(Fv_sorted)):
                x = Fv_sorted[i, 0]
                y = Fv_sorted[i, 1]
                if y < max_y and x < ref[0]:
                    # This point contributes a rectangle
                    # The width is from x to next point's x or ref[0]
                    next_x = Fv_sorted[i + 1, 0] if i + 1 < len(Fv_sorted) else ref[0]
                    width = min(next_x, ref[0]) - x
                    if width > 0:
                        hv += width * (max_y - y)
                    if y < max_y:
                        max_y = y
            return hv

        def hv_contribution(F, i, ref):
            """Compute hypervolume contribution of point i."""
            # Total HV without point i vs total HV with point i
            n = len(F)
            if n <= 1:
                return hypervolume_2d(F, ref)
            F_without = np.delete(F, i, axis=0)
            hv_with = hypervolume_2d(F, ref)
            hv_without = hypervolume_2d(F_without, ref)
            return hv_with - hv_without

        # Halton sequence for initial population
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            """Generate n points in [0,1]^dim using Halton sequences."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # 2, 3, 5, 7...
                for i in range(n):
                    points[i, d] = halton(i + 1, base)
            return points

        # Initial population using Halton
        pop_size = min(30, budget // 5)
        if pop_size < 10:
            pop_size = 10
        halton_pts = halton_sequence(pop_size, dim)
        X = lo + (hi - lo) * halton_pts
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Check deadline
        if time.time() > deadline:
            return X

        # Main loop: hypervolume-based selection with local search
        while used < budget:
            if time.time() > deadline:
                break

            # Get non-dominated set
            nd_mask = is_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]

            if len(nd_indices) == 0:
                # Fallback: use all points
                nd_indices = np.arange(len(F))

            # Compute hypervolume contributions for non-dominated points
            nd_F = F[nd_indices]
            nd_X = X[nd_indices]
            n_nd = len(nd_F)

            if n_nd == 0:
                # Should not happen
                break

            # Select parent based on hypervolume contribution
            # For efficiency, use a simpler selection when population is large
            if n_nd > 20:
                # Use random selection from non-dominated set
                parent_idx = nd_indices[np.random.randint(n_nd)]
            else:
                # Compute contributions
                contributions = []
                for k in range(n_nd):
                    # HV without this point
                    F_without = np.delete(nd_F, k, axis=0)
                    hv_without = hypervolume_2d(F_without, ref)
                    hv_with = hypervolume_2d(nd_F, ref)
                    contributions.append(hv_with - hv_without)
                contributions = np.array(contributions)
                # Select parent with highest contribution (or random among top)
                # Use tournament: pick 2, take better
                if len(contributions) > 1:
                    c1, c2 = np.random.choice(len(contributions), 2, replace=False)
                    parent_local = c1 if contributions[c1] > contributions[c2] else c2
                else:
                    parent_local = 0
                parent_idx = nd_indices[parent_local]

            # Local search around parent
            parent = X[parent_idx].copy()

            # Try multiple local search directions
            best_child = None
            best_child_f = None
            best_hv_gain = -np.inf

            # Use a few random perturbations
            n_local = 3
            for _ in range(n_local):
                if used >= budget or time.time() > deadline:
                    break
                # Random direction with decreasing step
                step = 0.05 * (hi - lo)
                direction = np.random.randn(dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                child = parent + step * direction
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1

                # Compute HV gain if we replace the worst point
                # For simplicity, check if child is non-dominated
                # If it is, compute HV contribution
                is_nd = True
                for j in range(len(F)):
                    if j != parent_idx and dominates(F[j], fc):
                        is_nd = False
                        break

                if is_nd:
                    # Compute HV contribution of child
                    F_test = np.delete(F, parent_idx, axis=0)
                    F_test = np.vstack([F_test, fc.reshape(1, -1)])
                    hv_test = hypervolume_2d(F_test, ref)
                    hv_current = hypervolume_2d(F, ref)
                    gain = hv_test - hv_current
                    if gain > best_hv_gain:
                        best_hv_gain = gain
                        best_child = child.copy()
                        best_child_f = fc
                else:
                    # Child is dominated, still might be useful
                    # Check if it's better than the point it would replace
                    # For now, skip dominated children in local search
                    pass

            # If no good child found, try a mutation
            if best_child is None:
                # Simple mutation
                step = 0.1 * (hi - lo)
                child = parent + step * np.random.randn(dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                best_child = child
                best_child_f = fc

            # Replace the point with lowest HV contribution
            if n_nd > 20:
                # Replace random non-dominated point
                replace_idx = nd_indices[np.random.randint(n_nd)]
            else:
                # Find point with lowest HV contribution
                if len(contributions) > 0:
                    min_contrib_local = np.argmin(contributions)
                    replace_idx = nd_indices[min_contrib_local]
                else:
                    replace_idx = parent_idx

            X[replace_idx] = best_child
            F[replace_idx] = best_child_f

        return X
# EVOLVE-BLOCK-END
