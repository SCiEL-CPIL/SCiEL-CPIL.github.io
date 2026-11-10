"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with differential mutation and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = min(40, self.budget // 5)
        if pop_size < 10:
            pop_size = 10

        # Initial population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1

            current_front = [i for i in range(n) if n_dom[i] == 0]
            fronts.append(current_front)
            k = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for solutions in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.ones(n)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                val_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
                if val_range > 0:
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / val_range
            return dist

        def tournament_select(F, fronts, crowd, k=2):
            """Tournament selection based on rank and crowding distance."""
            n = len(F)
            candidates = rng.integers(0, n, size=k)
            best = candidates[0]
            for c in candidates[1:]:
                # Compare rank (lower is better)
                rank_c = next(i for i, f in enumerate(fronts) if c in f)
                rank_b = next(i for i, f in enumerate(fronts) if best in f)
                if rank_c < rank_b:
                    best = c
                elif rank_c == rank_b:
                    # Compare crowding distance (higher is better)
                    cd_c = crowd[c]
                    cd_b = crowd[best]
                    if cd_c > cd_b:
                        best = c
            return best

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Crowding distance for each front
            crowd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for idx, sol_idx in enumerate(front):
                        crowd[sol_idx] = cd[idx]

            # Create offspring
            n_offspring = pop_size
            offspring_X = np.empty((n_offspring, dim))
            offspring_F = np.empty((n_offspring, n_obj))

            for i in range(n_offspring):
                if time.time() > deadline:
                    break

                # Select 3 parents for DE/rand/1/bin
                p1 = tournament_select(F, fronts, crowd)
                p2 = tournament_select(F, fronts, crowd)
                p3 = tournament_select(F, fronts, crowd)

                # Ensure distinct parents if possible
                attempts = 0
                while p2 == p1 or p3 == p1 or p2 == p3:
                    if attempts > 10:
                        break
                    if p2 == p1:
                        p2 = tournament_select(F, fronts, crowd)
                    elif p3 == p1:
                        p3 = tournament_select(F, fronts, crowd)
                    elif p2 == p3:
                        p3 = tournament_select(F, fronts, crowd)
                    attempts += 1

                # Differential mutation
                F1 = 0.5 + 0.5 * rng.random()  # F in [0.5, 1.0]
                v = X[p1] + F1 * (X[p2] - X[p3])

                # Binary crossover
                cr = 0.5
                mask = rng.random(dim) < cr
                if not np.any(mask):
                    jrand = rng.integers(dim)
                    mask[jrand] = True

                child = np.where(mask, v, X[p1])
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring_X[i] = child
                offspring_F[i] = fc

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            # Non-dominated sort on combined
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            next_X = np.empty((pop_size, dim))
            next_F = np.empty((pop_size, n_obj))
            next_idx = []
            for front in combined_fronts:
                if len(next_idx) + len(front) <= pop_size:
                    next_idx.extend(front)
                else:
                    # Need to select from this front based on crowding
                    remaining = pop_size - len(next_idx)
                    cd = crowding_distance(combined_F, front)
                    sorted_by_cd = np.argsort(-cd)
                    for si in sorted_by_cd[:remaining]:
                        next_idx.append(front[si])
                    break

            for i, idx in enumerate(next_idx):
                next_X[i] = combined_X[idx]
                next_F[i] = combined_F[idx]

            X = next_X
            F = next_F

        # Return final population
        return X
# EVOLVE-BLOCK-END
