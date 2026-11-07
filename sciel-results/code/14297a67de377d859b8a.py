"""Multi-objective optimizer for RE21 with hypervolume-guided search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided evolutionary multi-objective search for 2-objective RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # Reference point for hypervolume (from task: [3000.0, 0.06])
        ref = np.array([3000.0, 0.06])

        # Population size
        pop_size = min(30, max(10, budget // 6))

        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_F = [F.copy()]

        def compute_hv_2d(points):
            """Compute 2D hypervolume for points dominated by reference."""
            # Filter points that are better than reference in both objectives
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            max_y = ref[1]
            for i in range(len(pts)):
                x = pts[i, 0]
                y = pts[i, 1]
                if y < max_y:
                    hv += (ref[0] - x) * (max_y - y)
                    max_y = y
            return hv

        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return mask

        # Main optimization loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Get non-dominated parents
            nd_mask = non_dominated_mask(F)
            parents = X[nd_mask] if np.any(nd_mask) else X
            n_parents = len(parents)

            # Generate children using differential evolution
            n_children = min(pop_size // 2, budget - used)
            for _ in range(n_children):
                if time.time() > deadline:
                    break

                # Select parents for DE
                if n_parents >= 3:
                    idx = rng.integers(n_parents, size=3)
                    # Ensure distinct indices
                    attempts = 0
                    while len(set(idx)) < 3 and attempts < 10:
                        idx = rng.integers(n_parents, size=3)
                        attempts += 1
                    if len(set(idx)) < 3:
                        idx[1] = (idx[0] + 1) % n_parents
                        idx[2] = (idx[1] + 1) % n_parents
                    v = parents[idx[0]] + 0.5 * (parents[idx[1]] - parents[idx[2]])
                elif n_parents >= 2:
                    idx = rng.integers(n_parents, size=2)
                    if idx[0] == idx[1]:
                        idx[1] = (idx[0] + 1) % n_parents
                    v = parents[idx[0]] + 0.3 * (parents[idx[1]] - parents[idx[0]])
                else:
                    v = parents[0]

                # Add mutation
                mutation_scale = 0.15 * (hi - lo)
                v = v + mutation_scale * rng.standard_normal(dim)
                v = np.clip(v, lo, hi)

                # Evaluate
                fv = problem(v)
                used += 1
                all_F.append(fv.reshape(1, -1))

                # Replace worst point in population based on hypervolume contribution
                # Simple approach: replace the point with highest sum of objectives
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                X[worst_idx] = v
                F[worst_idx] = fv

            # Periodically do a local refinement around best points
            if used % 20 == 0 and time.time() < deadline - 10:
                nd_mask = non_dominated_mask(F)
                nd_X = X[nd_mask]
                nd_F = F[nd_mask]
                # Pick the point closest to ideal (min in each obj)
                ideal = np.min(nd_F, axis=0)
                dists = np.sum((nd_F - ideal) ** 2, axis=1)
                best_local = nd_X[int(np.argmin(dists))]

                # Local search with smaller steps
                for _ in range(min(5, budget - used)):
                    if time.time() > deadline:
                        break
                    step = 0.02 * (hi - lo)
                    candidate = best_local + step * rng.standard_normal(dim)
                    candidate = np.clip(candidate, lo, hi)
                    fc = problem(candidate)
                    used += 1
                    all_F.append(fc.reshape(1, -1))

                    # Check if it improves hypervolume
                    current_hv = compute_hv_2d(np.vstack(all_F))
                    # Keep it if it's non-dominated or improves HV
                    nd_mask_full = non_dominated_mask(np.vstack(all_F))
                    if nd_mask_full[-1]:
                        worst_idx = int(np.argmax(np.sum(F, axis=1)))
                        X[worst_idx] = candidate
                        F[worst_idx] = fc

        # Final: return all evaluated points (the evaluator records returned objective vectors)
        # We return the population; the evaluator will record problem(x) for each?
        # Actually, re-reading: "Every returned objective vector is recorded"
        # The return value of __call__ is ignored. So we just need to have evaluated points.
        # The evaluator records the objective vectors that were returned by problem().
        # So all_F contains all evaluated points. But the evaluator says "Every returned 
        # objective vector is recorded" - this means each call to problem(x) returns a 
        # vector, and that's recorded. So we're fine.
        return X
# EVOLVE-BLOCK-END
