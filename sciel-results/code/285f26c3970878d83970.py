# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s/330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Initial population
        pop_size = min(budget, 50)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Fast non-dominated sort
        def fast_nondominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            # Remove trailing empty front
            if not fronts[-1]:
                fronts.pop()
            return fronts

        # Crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.ones(len(indices))
            m = len(indices)
            cd = np.zeros(m)
            F_sub = F[indices]
            for k in range(n_obj):
                order = np.argsort(F_sub[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                if F_sub[order[-1], k] - F_sub[order[0], k] > 1e-15:
                    for i in range(1, m - 1):
                        cd[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / (F_sub[order[-1], k] - F_sub[order[0], k])
            return cd

        # Tournament selection
        def tournament_select(fronts, X, F, k=2):
            idx = list(range(len(F)))
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for i in front:
                    rank[i] = r
            best = 0
            for _ in range(k):
                cand = rng.integers(len(F))
                if rank[cand] < rank[best] or (rank[cand] == rank[best] and F[cand].sum() < F[best].sum()):
                    best = cand
            return X[best].copy()

        # Main evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            fronts = fast_nondominated_sort(F)

            # Select parents via tournament
            p1 = tournament_select(fronts, X, F)
            p2 = tournament_select(fronts, X, F)

            # Crossover (SBX-like blend)
            if rng.random() < 0.9:
                # Simple blend crossover
                alpha = rng.random()
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()

            # Mutation with adaptive step size
            step = 0.05 + 0.1 * rng.random()
            mask = rng.random(dim) < 0.3
            child[mask] += step * (hi - lo)[mask] * rng.standard_normal(int(mask.sum())) if mask.any() else 0
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst individual
            # Find dominated or worst by sum
            dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
            if np.any(dominated):
                dom_indices = np.where(dominated)[0]
                worst = dom_indices[rng.integers(len(dom_indices))]
            else:
                worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
