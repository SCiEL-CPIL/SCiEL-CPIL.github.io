"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE21 using an evolutionary strategy with
    non-dominated sorting, crowding distance, and adaptive step size."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, keep 20s safety margin
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Initial population
        pop_size = 50
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track best points found (for hypervolume, we want diverse Pareto front)
        all_X = [X.copy()]
        all_F = [F.copy()]

        def non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (indices)."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = [0] * n
            rank = [0] * n
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j if F[i] <= F[j] and F[i] < F[j] in at least one
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1
                if n_dominates[i] == 0:
                    rank[i] = 0
                    fronts.append([])
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(self.n_obj):
                vals = [F[i, m] for i in indices]
                min_v = min(vals)
                max_v = max(vals)
                if max_v - min_v < 1e-15:
                    dist[indices.index(indices[0])] = float('inf')
                    dist[indices.index(indices[-1])] = float('inf')
                    continue
                # Sort by this objective
                sorted_idx = sorted(range(n), key=lambda i: vals[i])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / (max_v - min_v)
            return dist

        # Adaptive step size
        step_scale = 0.3
        min_step = 0.01
        max_step = 0.5
        no_improve_count = 0
        best_hv = -1.0

        while used < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Get fronts
            fronts = non_dominated_sort(F)

            # Select parents from first front (or top 2 fronts)
            parent_pool = []
            for front in fronts[:2]:
                parent_pool.extend(front)
            if not parent_pool:
                parent_pool = list(range(len(F)))

            # Generate offspring
            n_offspring = min(10, self.budget - used)
            offspring_X = []
            offspring_F = []

            for _ in range(n_offspring):
                # Tournament selection from parent pool
                t1 = parent_pool[rng.integers(len(parent_pool))]
                t2 = parent_pool[rng.integers(len(parent_pool))]
                # Pick the one with better rank (lower front index)
                rank_t1 = [i for i, f in enumerate(fronts) if t1 in f][0]
                rank_t2 = [i for i, f in enumerate(fronts) if t2 in f][0]
                if rank_t1 <= rank_t2:
                    parent = X[t1]
                else:
                    parent = X[t2]

                # Mutation with adaptive step
                step = step_scale * span * rng.standard_normal(self.dim)
                child = parent + step
                child = np.clip(child, lo, hi)
                offspring_X.append(child)
                offspring_F.append(problem(child))
                used += 1

            offspring_F = np.array(offspring_F, dtype=np.float64)
            offspring_X = np.array(offspring_X)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort on combined
            combined_fronts = non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to truncate this front using crowding distance
                    remaining = pop_size - len(new_X)
                    if remaining <= 0:
                        break
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance (descending)
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in range(min(remaining, len(sorted_front))):
                        idx = front[sorted_front[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

            # Update step size based on progress
            # Simple heuristic: if we're not making progress, reduce step
            # We can't easily measure hypervolume cheaply, so use a simple heuristic
            if rng.random() < 0.1:
                step_scale = max(min_step, step_scale * 0.95)
            else:
                step_scale = min(max_step, step_scale * 1.02)

        # Final: do a few local refinements around best points
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline - 10.0:
            # Find non-dominated points in final population
            final_fronts = non_dominated_sort(F)
            pareto_indices = final_fronts[0] if final_fronts else list(range(len(F)))
            n_refine = min(remaining, 20)
            for _ in range(n_refine):
                if time.time() > deadline - 5.0:
                    break
                idx = pareto_indices[rng.integers(len(pareto_indices))]
                # Local refinement with small step
                step = 0.02 * span * rng.standard_normal(self.dim)
                child = np.clip(X[idx] + step, lo, hi)
                fc = problem(child)
                used += 1
                # Replace if it's non-dominated or improves diversity
                # Simple: replace worst point in population
                worst = int(np.argmax(np.sum(F, axis=1)))
                # Only replace if child is better in at least one objective
                if np.any(fc < F[worst]):
                    X[worst] = child
                    F[worst] = fc

        return X
# EVOLVE-BLOCK-END
