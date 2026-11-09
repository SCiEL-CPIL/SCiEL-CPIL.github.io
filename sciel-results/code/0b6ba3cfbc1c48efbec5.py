"""Multi-objective optimizer on RE34 using a population-based evolutionary approach with
hypervolume-aware selection and adaptive mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary optimizer with hypervolume-guided selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 325.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        rng = np.random.default_rng(42)

        # Initialize population
        pop_size = min(40, budget // 5)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front

            # Add any remaining (shouldn't happen in proper implementation)
            assigned = set()
            for front in fronts:
                for idx in front:
                    assigned.add(idx)
            if len(assigned) < n:
                remaining = [i for i in range(n) if i not in assigned]
                fronts.append(remaining)

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distances for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                min_val = vals.min()
                max_val = vals.max()
                if max_val - min_val < 1e-15:
                    continue
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_val - min_val)
            return dist

        def hypervolume_contribution(F, ref):
            """Compute contribution of each point to hypervolume (approximate)."""
            n = len(F)
            contribs = np.zeros(n)
            # Simple approximation: for each point, compute volume it uniquely dominates
            for i in range(n):
                vol = 1.0
                for m in range(n_obj):
                    diff = ref[m] - F[i, m]
                    if diff <= 0:
                        vol = 0.0
                        break
                    # Check if any other point dominates this region
                    vol *= diff
                contribs[i] = vol
            return contribs

        ref_point = np.array([1705.0, 12.0, 0.3], dtype=np.float64)

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Get non-dominated fronts
            fronts = fast_non_dominated_sort(F)

            # Select parents from first few fronts
            n_parents = max(2, min(8, pop_size // 4))
            parent_pool = []
            for front in fronts[:2]:
                parent_pool.extend(front)
                if len(parent_pool) >= n_parents:
                    break

            if not parent_pool:
                parent_pool = list(range(pop_size))

            # Select parents with tournament selection
            parents_idx = []
            for _ in range(n_parents):
                # Tournament of 2
                candidates = rng.choice(parent_pool, size=min(2, len(parent_pool)), replace=False)
                if len(candidates) == 1:
                    parents_idx.append(candidates[0])
                else:
                    # Pick the one with better hypervolume contribution
                    hv1 = hypervolume_contribution(F[candidates[0:1]], ref_point)[0]
                    hv2 = hypervolume_contribution(F[candidates[1:2]], ref_point)[0]
                    parents_idx.append(candidates[0] if hv1 >= hv2 else candidates[1])

            # Create children
            n_children = max(1, min(4, budget - used))
            for c in range(n_children):
                if used >= budget or time.time() > deadline:
                    break

                # Crossover: blend two parents
                if len(parents_idx) >= 2:
                    p1_idx = parents_idx[c % len(parents_idx)]
                    p2_idx = parents_idx[(c + 1) % len(parents_idx)]
                    alpha = rng.random()
                    child_x = alpha * X[p1_idx] + (1 - alpha) * X[p2_idx]
                else:
                    child_x = X[parents_idx[0]].copy()

                # Adaptive mutation: scale with remaining budget
                mutation_scale = 0.1 * (1.0 + 0.5 * (used / budget))
                noise = mutation_scale * span * rng.standard_normal(dim)
                child_x = child_x + noise
                child_x = np.clip(child_x, lo, hi)

                fc = problem(child_x)
                used += 1

                # Replace worst individual using hypervolume-based fitness
                # Use a simple metric: distance to reference point (lower is better for minimization)
                # But we want to maintain diversity, so use a combination
                worst_idx = self._find_worst_to_replace(F, ref_point, rng)
                X[worst_idx] = child_x
                F[worst_idx] = fc

        return X

    def _find_worst_to_replace(self, F, ref_point, rng):
        """Find the worst individual to replace using a combination of
        hypervolume contribution and dominance status."""
        n = len(F)
        n_obj = F.shape[1]

        # Check if point is dominated
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break

        # Prefer replacing dominated points
        if np.any(dominated):
            # Among dominated, pick one with worst hypervolume contribution
            dominated_indices = np.where(dominated)[0]
            # Use a simple score: sum of objective values (higher is worse)
            scores = F[dominated_indices].sum(axis=1)
            worst_local = np.argmax(scores)
            return dominated_indices[worst_local]

        # If no dominated points, replace the one with least hypervolume contribution
        # Approximate contribution as product of (ref - F[i])
        contribs = np.ones(n)
        for m in range(n_obj):
            contribs *= np.maximum(ref_point[m] - F[:, m], 0)

        # Add some randomness to avoid cycling
        noise = rng.random(n) * 0.1 * contribs.max() if contribs.max() > 0 else 0
        scores = -contribs + noise
        return int(np.argmax(scores))
# EVOLVE-BLOCK-END
