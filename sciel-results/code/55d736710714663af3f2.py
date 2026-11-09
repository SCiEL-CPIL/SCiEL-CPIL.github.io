"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    nd[i] += 1
        front = [i for i in range(n) if nd[i] == 0]
        if not front:
            front = [0]
        fronts.append(front)
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front
        return fronts

    def _crowding_distance(self, F, indices):
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        dist = np.zeros(n)
        F_sub = F[indices]
        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            min_val = F_sub[order[0], m]
            max_val = F_sub[order[-1], m]
            if max_val == min_val:
                continue
            for k in range(1, n - 1):
                dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / (max_val - min_val)
        return dist

    def _compute_ranks(self, F):
        """Return array of rank for each individual."""
        fronts = self._fast_non_dominated_sort(F)
        rank = np.zeros(len(F), dtype=int)
        for fi, front in enumerate(fronts):
            for idx in front:
                rank[idx] = fi
        return rank, fronts

    def _select(self, F, pop_size):
        rank, fronts = self._compute_ranks(F)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                need = pop_size - len(selected)
                dist = self._crowding_distance(F, front)
                order = np.argsort(dist)[::-1]
                selected.extend([front[i] for i in order[:need]])
                break
        return selected

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        pop_size = 30
        X = lo + span * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        while used < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Compute ranks once
            rank, fronts = self._compute_ranks(F)

            # Generate offspring
            offspring = []
            while len(offspring) < pop_size:
                # Tournament selection
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                if rank[i1] < rank[i2]:
                    p1, p2 = i1, i2
                elif rank[i2] < rank[i1]:
                    p1, p2 = i2, i1
                else:
                    # Same front: use crowding distance
                    # Find which front they belong to
                    for front in fronts:
                        if i1 in front and i2 in front:
                            dist = self._crowding_distance(F, front)
                            pos1 = front.index(i1)
                            pos2 = front.index(i2)
                            if dist[pos1] >= dist[pos2]:
                                p1, p2 = i1, i2
                            else:
                                p1, p2 = i2, i1
                            break
                    else:
                        p1, p2 = i1, i2

                # Crossover + mutation
                if np.random.random() < 0.9:
                    mask = np.random.random(self.dim) < 0.5
                    c1 = np.where(mask, X[p1], X[p2]).copy()
                    c2 = np.where(mask, X[p2], X[p1]).copy()
                    c1 += 0.1 * span * np.random.standard_normal(self.dim)
                    c2 += 0.1 * span * np.random.standard_normal(self.dim)
                    c1 = np.clip(c1, lo, hi)
                    c2 = np.clip(c2, lo, hi)
                else:
                    c1 = X[p1].copy() + 0.1 * span * np.random.standard_normal(self.dim)
                    c2 = X[p2].copy() + 0.1 * span * np.random.standard_normal(self.dim)
                    c1 = np.clip(c1, lo, hi)
                    c2 = np.clip(c2, lo, hi)
                offspring.append(c1)
                offspring.append(c2)

            offspring = np.array(offspring[:pop_size])
            F_off = np.array([problem(x) for x in offspring])
            used += pop_size

            if time.time() > deadline - 5.0:
                break

            # Combine and select
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, F_off])
            next_idx = self._select(F_all, pop_size)
            X = X_all[next_idx]
            F = F_all[next_idx]

        return X
# EVOLVE-BLOCK-END
