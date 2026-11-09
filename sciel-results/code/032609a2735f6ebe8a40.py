"""Multi-objective optimizer on WFG4 with island model and weight-based DE."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island-model multi-objective optimizer with weight-based DE for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _dominance_sort(F):
        """Return non-dominated sort fronts as list of lists of indices."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        n_dominated = np.zeros(n, dtype=int)
        fronts = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominated[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dominated[i] += 1
        front0 = [i for i in range(n) if n_dominated[i] == 0]
        fronts.append(front0)
        while front0:
            next_front = []
            for i in front0:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front0 = next_front
        return fronts

    @staticmethod
    def _crowding_distance(F, front_indices):
        """Compute crowding distance for indices in front."""
        m = len(front_indices)
        if m <= 2:
            return np.full(m, np.inf)
        dist = np.zeros(m)
        for k in range(F.shape[1]):
            vals = F[front_indices, k]
            v_min, v_max = vals.min(), vals.max()
            if v_max == v_min:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for idx in range(1, m - 1):
                dist[order[idx]] += (vals[order[idx + 1]] - vals[order[idx - 1]]) / (v_max - v_min)
        return dist

    @staticmethod
    def _hv_contribution(point, ref_point, other_points):
        """Estimate hypervolume contribution of a point given others and reference."""
        n_obj = len(ref_point)
        if n_obj == 1:
            if point[0] < ref_point[0]:
                return ref_point[0] - point[0]
            return 0.0
        if n_obj == 2:
            vol = 0.0
            for i in range(2):
                if point[i] < ref_point[i]:
                    other = 1 - i
                    max_other = ref_point[other]
                    for p in other_points:
                        if p[other] < max_other and p[i] <= point[i]:
                            max_other = min(max_other, p[other])
                    if point[i] < max_other:
                        vol += (ref_point[i] - point[i]) * (max_other - point[other])
            return vol
        # For 3+ objectives, use simple approximation
        if any(point[i] >= ref_point[i] for i in range(n_obj)):
            return 0.0
        # Simple box approximation
        box_vol = 1.0
        for i in range(n_obj):
            box_vol *= (ref_point[i] - point[i])
        return box_vol

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # Define islands with different weight vectors for biasing search
        n_islands = 6
        island_weights = []
        for i in range(n_islands):
            # Generate diverse weight vectors
            w = rng.random(n_obj)
            w = w / w.sum()
            island_weights.append(w)

        # Population size per island
        total_pop = min(budget, 180)
        pop_per_island = max(5, total_pop // n_islands)

        # Initialize islands
        islands_X = []
        islands_F = []
        used = 0

        for i in range(n_islands):
            X_i = lo + (hi - lo) * rng.random((pop_per_island, dim))
            F_i = np.array([problem(x) for x in X_i])
            used += pop_per_island
            islands_X.append(X_i)
            islands_F.append(F_i)

        # DE parameters
        CR = 0.7
        F_base = 0.5

        # Migration interval
        migrate_every = 15

        generation = 0

        while used < budget:
            if time.time() > deadline:
                break

            generation += 1

            for island_idx in range(n_islands):
                if used >= budget or time.time() > deadline:
                    break

                X_i = islands_X[island_idx]
                F_i = islands_F[island_idx]
                n_i = len(X_i)

                if n_i < 3:
                    continue

                # Get weight vector for this island
                w = island_weights[island_idx]

                # Compute weighted scalarization score for parent selection
                scores = F_i @ w

                # Select 3 distinct parents based on weighted score (better = lower)
                parent_order = np.argsort(scores)
                # Pick top 3 non-dominated or best weighted
                p1_idx = parent_order[0]
                # For p2, p3 pick from next best
                remaining = [i for i in parent_order if i != p1_idx]
                if len(remaining) >= 2:
                    p2_idx = remaining[0]
                    p3_idx = remaining[1]
                elif len(remaining) == 1:
                    p2_idx = remaining[0]
                    p3_idx = p1_idx
                else:
                    p2_idx = p1_idx
                    p3_idx = p1_idx

                # DE mutation with weight-specific intensity
                # Higher weight on an objective -> more exploration in that direction
                f_scale = F_base + 0.2 * np.max(w)  # Slightly higher F if one weight dominates
                v = X_i[p1_idx] + f_scale * (X_i[p2_idx] - X_i[p3_idx])

                # DE crossover
                jrand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = X_i[p1_idx][j]

                trial = np.clip(trial, lo, hi)

                # Evaluate
                ft = problem(trial)
                used += 1

                # Selection: replace worst point in island based on weighted score
                # Compute scores for all current points + trial
                scores_all = np.vstack([F_i, ft.reshape(1, -1)]) @ w
                # Find the worst (highest score) to replace
                worst_pos = np.argmax(scores_all)

                if worst_pos < n_i:
                    # Replace an existing point
                    X_i[worst_pos] = trial
                    F_i[worst_pos] = ft
                else:
                    # Trial is better than all; replace the overall worst
                    worst_existing = np.argmax(scores_all[:n_i])
                    X_i[worst_existing] = trial
                    F_i[worst_existing] = ft

                islands_X[island_idx] = X_i
                islands_F[island_idx] = F_i

            # Migration: exchange best individuals between islands
            if generation % migrate_every == 0:
                for i in range(n_islands):
                    if time.time() > deadline:
                        break
                    F_i = islands_F[i]
                    X_i = islands_X[i]
                    n_i = len(X_i)
                    if n_i == 0:
                        continue

                    # Find best point in island i by weighted score
                    w_i = island_weights[i]
                    scores_i = F_i @ w_i
                    best_idx_i = np.argmin(scores_i)

                    # Choose a neighbor island to migrate to
                    neighbor = (i + 1) % n_islands
                    F_n = islands_F[neighbor]
                    X_n = islands_X[neighbor]
                    n_n = len(X_n)
                    if n_n == 0:
                        continue

                    w_n = island_weights[neighbor]
                    scores_n = F_n @ w_n

                    # Compare: if migrated point is better than worst in neighbor, swap
                    # Use weighted score in neighbor's context
                    migrated_score = F_i[best_idx_i] @ w_n
                    worst_in_neighbor = np.argmax(scores_n)

                    if migrated_score < scores_n[worst_in_neighbor]:
                        # Migrate: send best from i to neighbor, get worst from neighbor back
                        temp_X = X_n[worst_in_neighbor].copy()
                        temp_F = F_n[worst_in_neighbor].copy()

                        X_n[worst_in_neighbor] = X_i[best_idx_i]
                        F_n[worst_in_neighbor] = F_i[best_idx_i]

                        X_i[best_idx_i] = temp_X
                        F_i[best_idx_i] = temp_F

                        islands_X[neighbor] = X_n
                        islands_F[neighbor] = F_n
                        islands_X[i] = X_i
                        islands_F[i] = F_i

        return None
# EVOLVE-BLOCK-END
