"""Multi-objective optimizer on DTLZ2 with time-budget DE-based search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer for DTLZ2 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        n_obj = self.n_obj
        dim = self.dim

        # Population size
        pop_size = 60

        # Initialize population uniformly
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            front = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if dominates(F[i], F[j]):
                            S[i].append(j)
                        elif dominates(F[j], F[i]):
                            n_count[i] += 1
                if n_count[i] == 0:
                    front.append(i)

            fronts = []
            while front:
                fronts.append(front[:])
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                front = next_front
            return fronts

        def crowding_distance(F, indices):
            n = len(indices)
            if n == 0:
                return np.array([])
            dist = np.zeros(n)
            for m in range(n_obj):
                sorted_idx = np.argsort(F[indices, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_min = F[indices[sorted_idx[0]], m]
                f_max = F[indices[sorted_idx[-1]], m]
                if f_max > f_min:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (F[indices[sorted_idx[k + 1]], m] - F[indices[sorted_idx[k - 1]], m]) / (f_max - f_min)
            return dist

        # Main loop: use DE mutation with best/1/bin strategy
        while used < 200:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # For each individual, try to generate a better replacement
            # Use DE best/1/bin: v = x_best + F*(x_r1 - x_r2)
            F_factor = 0.8 + 0.2 * rng.random()
            CR = 0.5

            # Get first front (best)
            if len(fronts) == 0 or len(fronts[0]) == 0:
                break

            best_front = fronts[0]
            n_best = len(best_front)

            # Generate candidates for replacement
            candidates = []
            for _ in range(min(5, pop_size - used)):
                if used >= 200:
                    break
                if time.time() > deadline:
                    break

                # Select base from best front
                base_idx = best_front[rng.integers(n_best)]

                # Select two random distinct individuals
                r1 = rng.integers(pop_size)
                r2 = rng.integers(pop_size)
                while r2 == r1:
                    r2 = rng.integers(pop_size)

                # DE mutation
                v = X[base_idx] + F_factor * (X[r1] - X[r2])

                # Binomial crossover
                child = X[base_idx].copy()
                for d in range(dim):
                    if rng.random() < CR:
                        child[d] = v[d]

                # Ensure at least one gene is from mutant
                if np.all(child == X[base_idx]):
                    d_rand = rng.integers(dim)
                    child[d_rand] = v[d_rand]

                # Clip to bounds
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                candidates.append((child, fc))

            if not candidates:
                break

            # Combine population with candidates and select
            X_combined = np.vstack([X] + [c[0].reshape(1, -1) for c in candidates])
            F_combined = np.vstack([F] + [c[1].reshape(1, -1) for c in candidates])

            # Non-dominated sort the combined population
            fronts_combined = fast_non_dominated_sort(F_combined)

            # Build new population of size pop_size
            X_new = np.zeros((pop_size, dim))
            F_new = np.zeros((pop_size, n_obj))
            count = 0

            for front in fronts_combined:
                if count + len(front) <= pop_size:
                    for idx in front:
                        X_new[count] = X_combined[idx]
                        F_new[count] = F_combined[idx]
                        count += 1
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - count
                    cd = crowding_distance(F_combined, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        X_new[count] = X_combined[idx]
                        F_new[count] = F_combined[idx]
                        count += 1
                    break
                if count >= pop_size:
                    break

            if count < pop_size:
                # Fill remaining with random points
                for i in range(count, pop_size):
                    x_rand = lo + (hi - lo) * rng.random(dim)
                    f_rand = problem(x_rand)
                    used += 1
                    X_new[i] = x_rand
                    F_new[i] = f_rand

            X = X_new
            F = F_new

        return X
# EVOLVE-BLOCK-END
