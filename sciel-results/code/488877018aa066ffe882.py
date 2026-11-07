"""Multi-objective optimizer on DTLZ2 with NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        pop_size = 30
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return fronts as list of lists of indices."""
            n = len(F)
            if n == 0:
                return []
            S = [[] for _ in range(n)]
            nd = [0] * n
            rank = [0] * n
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        nd[i] += 1
                if nd[i] == 0:
                    rank[i] = 0
                    fronts.append([i])

            k = 0
            while k < len(fronts):
                current_front = fronts[k]
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for indices in a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(self.n_obj):
                sorted_idx = sorted(range(n), key=lambda i: F[front_indices[i], m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                f_min = F[front_indices[sorted_idx[0]], m]
                f_max = F[front_indices[sorted_idx[-1]], m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (
                        F[front_indices[sorted_idx[i + 1]], m] -
                        F[front_indices[sorted_idx[i - 1]], m]
                    ) / f_range
            return dist

        while used < self.budget:
            if time.time() > deadline - 5:
                break

            fronts = fast_non_dominated_sort(F)

            # Tournament selection
            def tournament_select(fronts, F, k=2):
                idx1 = rng.randint(0, pop_size)
                idx2 = rng.randint(0, pop_size)
                # Find ranks
                rank1 = next((i for i, f in enumerate(fronts) if idx1 in f), 0)
                rank2 = next((i for i, f in enumerate(fronts) if idx2 in f), 0)
                if rank1 < rank2:
                    return idx1
                elif rank2 < rank1:
                    return idx2
                # Same rank, use crowding distance
                front_idx = rank1
                front = fronts[front_idx]
                cd = crowding_distance(F, front)
                pos1 = front.index(idx1)
                pos2 = front.index(idx2)
                if cd[pos1] >= cd[pos2]:
                    return idx1
                else:
                    return idx2

            # Generate offspring using SBX and mutation
            n_offspring = min(pop_size, self.budget - used)
            offspring_X = []
            offspring_F = []

            eta_c = 20.0
            eta_m = 20.0
            pm = 1.0 / self.dim

            for _ in range(n_offspring):
                p1 = tournament_select(fronts, F)
                p2 = tournament_select(fronts, F)
                if p1 == p2:
                    p2 = (p2 + 1) % pop_size

                # SBX crossover
                child = np.empty(self.dim)
                for d in range(self.dim):
                    if rng.random() < 0.5:
                        x1, x2 = X[p1, d], X[p2, d]
                    else:
                        x1, x2 = X[p2, d], X[p1, d]

                    if rng.random() <= 0.5:
                        if abs(x1 - x2) < 1e-14:
                            child[d] = x1
                        else:
                            if x1 < x2:
                                u = rng.random()
                                if u <= 1.0 / (eta_c + 1):
                                    beta = (2 * u) ** (1.0 / (eta_c + 1))
                                else:
                                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1))
                                c1 = 0.5 * ((x1 + x2) - beta * (x2 - x1))
                            else:
                                u = rng.random()
                                if u <= 1.0 / (eta_c + 1):
                                    beta = (2 * u) ** (1.0 / (eta_c + 1))
                                else:
                                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1))
                                c1 = 0.5 * ((x1 + x2) + beta * (x2 - x1))
                            child[d] = c1
                    else:
                        child[d] = x1

                # Polynomial mutation
                for d in range(self.dim):
                    if rng.random() < pm:
                        x = child[d]
                        if hi[d] - lo[d] < 1e-14:
                            continue
                        u = rng.random()
                        if u < 0.5:
                            xy = (x - lo[d]) / (hi[d] - lo[d])
                            delta1 = (2 * xy) ** (1.0 / (eta_m + 1))
                            delta = 1.0 - delta1
                        else:
                            xy = (hi[d] - x) / (hi[d] - lo[d])
                            delta2 = (2 * xy) ** (1.0 / (eta_m + 1))
                            delta = delta2 - 1.0
                        child[d] = x + delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                fchild = problem(child)
                offspring_X.append(child)
                offspring_F.append(fchild)
                used += 1
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F, dtype=np.float64)
            offspring_X = np.array(offspring_X, dtype=np.float64)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort on combined
            fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for i in front:
                        next_X.append(combined_X[i])
                        next_F.append(combined_F[i])
                else:
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_by_cd = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in sorted_by_cd[:remaining]:
                        next_X.append(combined_X[front[i]])
                        next_F.append(combined_F[front[i]])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
