"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II-like approach for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Initial population
        pop_size = min(50, self.budget // 2)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort returning list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominated = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominated[i] += 1

                if n_dominated[i] == 0:
                    fronts[0].append(i)

            i = 0
            while len(fronts[i]) > 0:
                next_front = []
                for k in fronts[i]:
                    for m in S[k]:
                        n_dominated[m] -= 1
                        if n_dominated[m] == 0:
                            next_front.append(m)
                i += 1
                if len(next_front) > 0:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            distances = np.zeros(n)
            F_sub = F[indices]

            for m in range(self.n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 0:
                    for k in range(1, n - 1):
                        distances[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / f_range

            return distances

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Sort by non-dominance and crowding
            fronts = fast_non_dominated_sort(F)

            # Select parents from first fronts
            parent_indices = []
            for front in fronts:
                if len(parent_indices) + len(front) > pop_size:
                    # Use crowding distance to select
                    cd = crowding_distance(F, front)
                    selected = front[np.argsort(-cd)[:pop_size - len(parent_indices)]]
                    parent_indices.extend(selected)
                    break
                else:
                    parent_indices.extend(front)

            # Generate offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size:
                if time.time() > deadline:
                    break

                # Binary tournament selection
                p1 = parent_indices[rng.integers(len(parent_indices))]
                p2 = parent_indices[rng.integers(len(parent_indices))]
                # Simple dominance check for tournament
                if np.all(F[p1] <= F[p2]) and np.any(F[p1] < F[p2]):
                    parent = p1
                elif np.all(F[p2] <= F[p1]) and np.any(F[p2] < F[p1]):
                    parent = p2
                else:
                    parent = p1 if rng.random() < 0.5 else p2

                # SBX crossover
                if rng.random() < 0.9:
                    p1_idx = parent_indices[rng.integers(len(parent_indices))]
                    if rng.random() < 0.5:
                        parent = p1_idx
                    # Simple mutation only for simplicity
                    child = X[parent].copy()
                    # Polynomial mutation
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            spread = 20
                            rand = rng.random()
                            if rand <= 0.5:
                                delta_q = (2 * rand) ** (1.0 / (spread + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - rand)) ** (1.0 / (spread + 1))
                            child[d] += delta_q * (hi[d] - lo[d])
                    child = np.clip(child, lo, hi)
                else:
                    child = X[parent].copy()
                    # Gaussian mutation
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            child[d] += 0.1 * (hi[d] - lo[d]) * rng.standard_normal()
                    child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])

            # Sort combined and select next generation
            fronts = fast_non_dominated_sort(F_combined)

            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    next_X.append(X_combined[front])
                    next_F.append(F_combined[front])
                else:
                    cd = crowding_distance(F_combined, front)
                    needed = pop_size - len(next_X)
                    selected = front[np.argsort(-cd)[:needed]]
                    next_X.append(X_combined[selected])
                    next_F.append(F_combined[selected])
                    break

            X = np.vstack(next_X) if next_X else X_combined[:pop_size]
            F = np.vstack(next_F) if next_F else F_combined[:pop_size]

        return X
# EVOLVE-BLOCK-END
