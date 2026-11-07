"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        pop_size = 40
        budget = self.budget

        # Initial population
        rng = np.random.default_rng(42)
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        if used >= budget:
            return X

        def fast_non_dominated_sort(F):
            """Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominates = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1

            current_front = [i for i in range(n) if n_dominates[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                current_front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for points in front_indices."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            distances = np.zeros(n)
            F_front = F[front_indices]

            for m in range(self.n_obj):
                sorted_idx = np.argsort(F_front[:, m])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                f_range = F_front[sorted_idx[-1], m] - F_front[sorted_idx[0], m]
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (
                        F_front[sorted_idx[i + 1], m] - F_front[sorted_idx[i - 1], m]
                    ) / f_range

            return distances

        while used < budget and time.time() < deadline:
            # Evaluate offspring
            n_offspring = min(pop_size, budget - used)
            if n_offspring <= 0:
                break

            # Selection: binary tournament on non-dominated ranking
            fronts = fast_non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            # Compute crowding distances per front
            cd = np.zeros(len(F))
            for front in fronts:
                cd[front] = crowding_distance(F, front)

            offspring_X = []
            offspring_F = []

            for _ in range(n_offspring):
                # Binary tournament
                i = rng.integers(len(F))
                j = rng.integers(len(F))
                if rank[i] < rank[j]:
                    p1 = i
                elif rank[j] < rank[i]:
                    p1 = j
                else:
                    p1 = i if cd[i] >= cd[j] else j

                i = rng.integers(len(F))
                j = rng.integers(len(F))
                if rank[i] < rank[j]:
                    p2 = i
                elif rank[j] < rank[i]:
                    p2 = j
                else:
                    p2 = i if cd[i] >= cd[j] else j

                # SBX crossover
                if rng.random() < 0.9:
                    child1 = np.zeros(self.dim)
                    child2 = np.zeros(self.dim)
                    for k in range(self.dim):
                        if rng.random() < 0.5:
                            if abs(X[p1, k] - X[p2, k]) < 1e-12:
                                child1[k] = child2[k] = X[p1, k]
                            else:
                                beta1 = (0.5 * (X[p1, k] / X[p2, k] if X[p2, k] != 0 else 1)) ** (
                                    1 / (2 * 15 + 2)
                                )
                                beta2 = (0.5 * (X[p2, k] / X[p1, k] if X[p1, k] != 0 else 1)) ** (
                                    1 / (2 * 15 + 2)
                                )
                                child1[k] = 0.5 * ((1 + beta1) * X[p1, k] + (1 - beta1) * X[p2, k])
                                child2[k] = 0.5 * ((1 + beta2) * X[p2, k] + (1 - beta2) * X[p1, k])
                        else:
                            child1[k] = X[p1, k]
                            child2[k] = X[p2, k]
                else:
                    child1 = X[p1].copy()
                    child2 = X[p2].copy()

                # Polynomial mutation
                for k in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        delta1 = (1 + 2 * max(0, (child1[k] - lo[k]) / span[k])) ** (-16)
                        delta2 = (1 + 2 * max(0, (hi[k] - child1[k]) / span[k])) ** (-16)
                        r = rng.random()
                        if r < 0.5:
                            child1[k] += span[k] * delta1
                        else:
                            child1[k] -= span[k] * delta2
                    if rng.random() < 1.0 / self.dim:
                        delta1 = (1 + 2 * max(0, (child2[k] - lo[k]) / span[k])) ** (-16)
                        delta2 = (1 + 2 * max(0, (hi[k] - child2[k]) / span[k])) ** (-16)
                        r = rng.random()
                        if r < 0.5:
                            child2[k] += span[k] * delta1
                        else:
                            child2[k] -= span[k] * delta2

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                offspring_X.append(child1)
                offspring_X.append(child2)
                offspring_F.append(problem(child1))
                offspring_F.append(problem(child2))
                used += 2

                if used >= budget:
                    break

                # Check deadline
                if time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X[:budget - used + len(offspring_X)])
            # Actually, we need to be careful with budget
            # Let's re-evaluate: we added up to n_offspring*2 evaluations
            # But we broke early, so let's just use what we have

            # Combine population and offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])

            # Truncate to pop_size using non-dominated sorting + crowding distance
            fronts = fast_non_dominated_sort(F_combined)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    remaining = pop_size - len(selected)
                    cd_front = crowding_distance(F_combined, front)
                    sorted_front = [front[i] for i in np.argsort(-cd_front)]
                    selected.extend(sorted_front[:remaining])
                    break

            X = X_combined[selected]
            F = F_combined[selected]

        return X
# EVOLVE-BLOCK-END
