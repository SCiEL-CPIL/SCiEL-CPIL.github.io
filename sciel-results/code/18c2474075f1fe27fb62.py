"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using efficient NSGA-II with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        d = self.dim
        n_obj = self.n_obj

        # Smaller population for speed, use all budget
        pop_size = 20
        X = lo + range_ * rng.random((pop_size, d))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return [[]]
            # Vectorized domination
            # F[i] dominates F[j] if all(F[i] <= F[j]) and any(F[i] < F[j])
            S = [set() for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_dom[i] += 1
            fronts = []
            current = [i for i in range(n) if n_dom[i] == 0]
            while current:
                fronts.append(current)
                next_front = []
                for i in current:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current = next_front
            return fronts

        def crowding_distance(F_subset):
            n = len(F_subset)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(F_subset.shape[1]):
                order = np.argsort(F_subset[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rng_m = F_subset[order[-1], m] - F_subset[order[0], m]
                if rng_m > 1e-15:
                    for i in range(1, n - 1):
                        cd[order[i]] += (F_subset[order[i + 1], m] - F_subset[order[i - 1], m]) / rng_m
            return cd

        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break

            fronts = fast_non_dominated_sort(F)
            if not fronts:
                break

            # Compute crowding distances for selection
            cd_all = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    F_front = F[front]
                    cd_front = crowding_distance(F_front)
                    for idx, c in zip(front, cd_front):
                        cd_all[idx] = c

            # Generate offspring: one at a time to stay within budget
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, self.budget - used)

            for _ in range(n_offspring):
                if time.time() > deadline:
                    break
                if used + len(offspring_F) >= self.budget:
                    break

                # Binary tournament: prefer non-dominated, then higher crowding
                i1 = rng.integers(len(F))
                i2 = rng.integers(len(F))
                # Prefer lower front rank (front index)
                front_rank = np.zeros(len(F), dtype=int)
                for fi, front in enumerate(fronts):
                    for idx in front:
                        front_rank[idx] = fi

                if front_rank[i1] < front_rank[i2]:
                    p1 = i1
                elif front_rank[i2] < front_rank[i1]:
                    p1 = i2
                else:
                    p1 = i1 if cd_all[i1] >= cd_all[i2] else i2

                # Mutation: Gaussian with adaptive sigma
                sigma = 0.15
                child = X[p1] + sigma * range_ * rng.standard_normal(d)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if not offspring_X:
                break

            # Combine and select
            X_comb = np.vstack([X, np.array(offspring_X)])
            F_comb = np.vstack([F, np.array(offspring_F)])
            fronts_comb = fast_non_dominated_sort(F_comb)

            new_X = []
            new_F = []
            for front in fronts_comb:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                else:
                    F_front = F_comb[front]
                    cd_front = crowding_distance(F_front)
                    order = np.argsort(-cd_front)
                    needed = pop_size - len(new_X)
                    for i in range(needed):
                        idx = front[order[i]]
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        # Final polish: refine the best non-dominated points with smaller steps
        if time.time() < deadline - 5 and used < self.budget:
            fronts_final = fast_non_dominated_sort(F)
            if fronts_final and fronts_final[0]:
                best_front = fronts_final[0]
                # Try small perturbations around best points
                for idx in best_front[:5]:
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    for _ in range(3):
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        child = X[idx] + 0.05 * range_ * rng.standard_normal(d)
                        child = np.clip(child, lo, hi)
                        fc = problem(child)
                        used += 1
                        # Replace if better in at least one objective and not worse in others
                        if np.all(fc <= F[idx]) and np.any(fc < F[idx]):
                            X[idx] = child
                            F[idx] = fc

        return X
# EVOLVE-BLOCK-END
