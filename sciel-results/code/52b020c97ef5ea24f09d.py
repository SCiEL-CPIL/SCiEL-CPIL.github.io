"""Rewritten optimizer for WFG9 (3 objectives, 12 dims) with NSGA-II and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer for WFG9 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = 40
        # Reserve some budget for potential local polish
        polish_budget = 20
        evolve_budget = self.budget - polish_budget

        # Initialize population
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Non-dominated sorting
        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting (Deb et al. 2002)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_count = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = []
            current_front = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_count[i] += 1
                if n_count[i] == 0:
                    current_front.append(i)

            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                current_front = next_front

            return fronts

        # Crowding distance
        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)

            distances = np.zeros(len(indices))
            n = len(indices)

            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = np.argsort(F[indices, m])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf

                f_min = F[indices[sorted_idx[0]], m]
                f_max = F[indices[sorted_idx[-1]], m]
                if f_max - f_min < 1e-15:
                    continue

                for i in range(1, n - 1):
                    prev = indices[sorted_idx[i - 1]]
                    curr = indices[sorted_idx[i]]
                    nxt = indices[sorted_idx[i + 1]]
                    distances[sorted_idx[i]] += (F[nxt, m] - F[prev, m]) / (f_max - f_min)

            return distances

        # Tournament selection
        def tournament_selection(F, fronts, k=2):
            """Select one individual using tournament selection."""
            idx1 = rng.integers(len(F))
            idx2 = rng.integers(len(F))

            # Find front numbers
            front_num1 = -1
            front_num2 = -1
            for fi, front in enumerate(fronts):
                if idx1 in front:
                    front_num1 = fi
                    break
                if idx2 in front:
                    front_num2 = fi
                    if front_num1 != -1:
                        break

            if front_num1 != front_num2:
                if front_num1 < front_num2:
                    return idx1
                else:
                    return idx2

            # Same front, use crowding distance
            front_indices = fronts[front_num1]
            cd = crowding_distance(F, front_indices)
            cd_map = dict(zip(front_indices, cd))
            cd1 = cd_map[idx1]
            cd2 = cd_map[idx2]

            if cd1 > cd2:
                return idx1
            else:
                return idx2

        # Crossover
        def crossover(p1, p2, cx_rate=0.9):
            """BLX-alpha style crossover."""
            if rng.random() > cx_rate:
                return p1.copy(), p2.copy()

            alpha = 0.5
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)

            for i in range(dim):
                low = min(p1[i], p2[i])
                high = max(p1[i], p2[i])
                span = high - low

                child1[i] = low + rng.random() * (1 + alpha) * span
                child2[i] = high - rng.random() * (1 + alpha) * span

                child1[i] = np.clip(child1[i], lo[i], hi[i])
                child2[i] = np.clip(child2[i], lo[i], hi[i])

            return child1, child2

        # Mutation
        def mutate(x, sigma=0.1):
            """Gaussian mutation."""
            x_new = x + sigma * (hi - lo) * rng.standard_normal(dim)
            return np.clip(x_new, lo, hi)

        # Main evolutionary loop
        sigma = 0.15
        cx_rate = 0.9

        while used < evolve_budget:
            if time.time() > deadline:
                break

            # Compute non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Generate offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size:
                if time.time() > deadline:
                    break

                # Select parents
                p1_idx = tournament_selection(F, fronts)
                p2_idx = tournament_selection(F, fronts)

                # Crossover
                child1, child2 = crossover(X[p1_idx], X[p2_idx], cx_rate)

                # Mutate
                child1 = mutate(child1, sigma)
                child2 = mutate(child2, sigma)

                # Evaluate
                f1 = problem(child1)
                used += 1
                if used > self.budget:
                    break

                offspring_X.append(child1)
                offspring_F.append(f1)

                if used >= self.budget or time.time() > deadline:
                    break

                f2 = problem(child2)
                used += 1
                if used > self.bbudget:
                    break

                offspring_X.append(child2)
                offspring_F.append(f2)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort the combined population
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(new_X)
                    if remaining <= 0:
                        break
                    cd = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(-cd)
                    for i in range(min(remaining, len(front))):
                        idx = front[sorted_front[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)
            else:
                break

            # Decay sigma
            sigma *= 0.995

        # Local polish phase
        if used < self.budget and len(F) > 0:
            # Find non-dominated points
            nd_front = fast_non_dominated_sort(F)
            nd_indices = nd_front[0] if nd_front else []

            if nd_indices:
                # Polish top few points
                num_polish = min(5, len(nd_indices))
                for idx in nd_indices[:num_polish]:
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                    p = X[idx].copy()
                    local_sigma = 0.05

                    for _ in range(3):
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break

                        candidate = p + local_sigma * (hi - lo) * rng.standard_normal(dim)
                        candidate = np.clip(candidate, lo, hi)
                        fc = problem(candidate)
                        used += 1

                        # Check if non-dominated w.r.t current population
                        is_dominated = False
                        for j in range(len(F)):
                            if np.all(F[j] <= fc) and np.any(F[j] < fc):
                                is_dominated = True
                                break

                        if not is_dominated:
                            # Add to population, replacing worst if needed
                            if len(X) < pop_size:
                                X = np.vstack([X, [candidate]])
                                F = np.vstack([F, [fc]])
                            else:
                                # Replace the dominated point with highest sum
                                sums = np.sum(F, axis=1)
                                worst_idx = np.argmax(sums)
                                if np.sum(fc) < sums[worst_idx]:
                                    X[worst_idx] = candidate
                                    F[worst_idx] = fc

                            p = candidate
                            local_sigma *= 0.7

        return X
# EVOLVE-BLOCK-END
