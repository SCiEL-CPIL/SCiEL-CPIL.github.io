"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid multi-objective optimizer: random init, non-dominated mutation, hypervolume-based replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1.1, 1.3, 1.1])

        # Initial population
        pop_size = min(20, self.budget // 4)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: find non-dominated indices
        def find_nondominated_indices(f):
            n = len(f)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(f[j] <= f[i]) and np.any(f[j] < f[i]):
                        nd[i] = False
            return np.where(nd)[0]

        # Helper: compute hypervolume contribution of a point given the set of points strictly better than reference
        def hypervolume_contribution(point, others):
            """
            Compute the hypervolume uniquely dominated by 'point' but not by any point in 'others'.
            For 3D, this is done by checking the region [point, ref] minus regions dominated by others.
            We use a simple decomposition: the volume is the product of (ref[i] - point[i]) for each dim,
            but we must subtract volumes dominated by other points that are 'inside' this box.
            However, a simpler and robust approach for small populations is to check if the point is
            non-dominated w.r.t. the set of points that are also better than ref.
            If it is dominated by any point in that set, its contribution is 0.
            If not, its contribution is not simply the full box volume because other points might
            cut into it. But for replacement selection, we can approximate by checking if it's non-dominated.
            Actually, for accurate replacement, we should calculate the *marginal* hypervolume.
            A simpler heuristic that works well: replace the point with the lowest hypervolume contribution.
            To avoid expensive exact HV calculation for every candidate vs every stored point,
            we can use the following:
            1. Filter F to only points strictly better than ref.
            2. Find non-dominated set among those.
            3. Among the dominated points (or all points), pick the one that is dominated by the most others,
               or simply the one with the highest sum of objectives (as a proxy for being 'worse').
            The current program uses sum of objectives. This is a valid heuristic.
            Let's stick with a slightly refined sum: normalized sum.
            """
            pass

        def get_replacement_index(F, new_point):
            """
            Decide which point to replace.
            Strategy:
            1. If new_point is dominated by any existing point, it's not great, but we still have to add it.
               We should replace the worst point.
            2. If new_point dominates some existing points, we should replace one of those.
            3. If new_point is non-dominated, we should replace the point with the lowest diversity/contribution.
            
            Simple robust heuristic:
            - Identify points that are strictly worse than ref (in any objective). These are useless for HV.
              If any exist, replace one of them (the worst one).
            - If all points are better than ref, replace the point with the highest sum of normalized objectives.
            """
            n = len(F)
            # Check if new point is better than ref
            new_better = np.all(new_point < ref)
            
            # Find indices of points that are NOT better than ref
            bad_indices = [i for i in range(n) if not np.all(F[i] < ref)]
            
            if bad_indices:
                # Among bad points, replace the one with the highest sum (worst)
                # Or just replace the first one? Replacing the worst is better.
                sums = np.sum(F[bad_indices], axis=1)
                worst_local = int(np.argmax(sums))
                return bad_indices[worst_local]
            else:
                # All points are good. Replace the one with the highest sum of objectives.
                # This tends to remove points that are 'far' from the origin, which might be less useful
                # if the front is convex, but for hypervolume, it's a reasonable proxy for 'dominatedness'
                # or low contribution.
                sums = np.sum(F, axis=1)
                return int(np.argmax(sums))

        while used < self.budget:
            # Time check
            if time.time() > deadline:
                break

            # Find non-dominated parents
            nd_idx = find_nondominated_indices(F)
            parents = X[nd_idx] if len(nd_idx) > 0 else X

            # Create 1 child per iteration (keep it simple and fast)
            # Select parent
            parent = parents[rng.integers(len(parents))].copy()

            # Adaptive mutation: start with larger steps, maybe shrink?
            # For now, fixed sigma = 0.1 * (hi - lo)
            sigma = 0.1 * (hi - lo)
            child = parent + sigma * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point
            worst_idx = get_replacement_index(F, fc)
            X[worst_idx], F[worst_idx] = child, fc

        return X
# EVOLVE-BLOCK-END
