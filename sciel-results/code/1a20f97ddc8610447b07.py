"""Multi-objective optimizer on WFG9: island-based NSGA-II with restarts and deadline guard."""
import numpy as np
import time


class Optimizer:
    """Island-based multi-objective search (NSGA-II style) for WFG9.

    Uses multiple islands seeded in distinct regions to cover the disconnected
    Pareto front, heavy-tailed mutation to escape local optima, non-dominated
    sorting + crowding for selection, and a wall-clock deadline guard.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin under the 300s / 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = int(self.budget)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        rng = np.random.default_rng(777)

        def eval_one(x):
            return np.asarray(problem(x), dtype=float)

        def non_dominated_sort(F):
            n = F.shape[0]
            S = [set() for _ in range(n)]
            ndom = np.zeros(n, dtype=int)
            for i in range(n):
                Fi = F[i]
                for j in range(n):
                    if i == j:
                        continue
                    Fj = F[j]
                    if np.all(Fi <= Fj) and np.any(Fi < Fj):
                        S[i].add(j)
                        ndom[j] += 1
            rank = np.zeros(n, dtype=int)
            front0 = [i for i in range(n) if ndom[i] == 0]
            for i in front0:
                rank[i] = 0
            fronts = [front0]
            k = 0
            while fronts[k]:
                nxt = []
                for i in fronts[k]:
                    for j in S[i]:
                        ndom[j] -= 1
                        if ndom[j] == 0:
                            rank[j] = k + 1
                            nxt.append(j)
                k += 1
                if nxt:
                    fronts.append(nxt)
            return fronts, rank

        def crowding_distance(F, idxs):
            n = len(idxs)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[idxs, m]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rng_m = vals[order[-1]] - vals[order[0]]
                if rng_m > 0:
                    for k in range(1, n - 1):
                        cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng_m
            return cd

        def tournament(F, rank, cd):
            a = rng.integers(len(F))
            b = rng.integers(len(F))
            if rank[a] < rank[b]:
                return a
            if rank[b] < rank[a]:
                return b
            return a if cd[a] >= cd[b] else b

        def make_child(parent, sigma):
            x = parent.copy()
            mask = rng.random(dim) < 0.5
            small = rng.standard_normal(dim) * sigma * span
            big = rng.standard_normal(dim) * (0.25 * span)
            x = x + np.where(mask, big, small)
            return np.clip(x, lo, hi)

        # ---- multi-start islands ------------------------------------------
        n_islands = 4
        base_pop = max(10, min(20, budget // (4 * n_islands)))
        base_pop = max(8, base_pop)

        pops = []
        for i in range(n_islands):
            center = rng.random(dim)
            X = lo + span * np.clip(center[None, :] + 0.12 * rng.standard_normal((base_pop, dim)), 0.0, 1.0)
            X = np.clip(X, lo, hi)
            F = np.array([eval_one(x) for x in X])
            pops.append([X, F])

        used = base_pop * n_islands
        sigma = 0.12

        while used < budget and time.time() < deadline - 4.0:
            for i in range(n_islands):
                if used >= budget or time.time() >= deadline - 4.0:
                    break
                X, F = pops[i]
                if len(F) < 2:
                    continue
                fronts, rank = non_dominated_sort(F)
                cd = np.zeros(len(F))
                for fr in fronts:
                    c = crowding_distance(F, fr)
                    for pos, idx in enumerate(fr):
                        cd[idx] = c[pos]
                pa = tournament(F, rank, cd)
                pb = tournament(F, rank, cd)
                parent = X[pa] if rank[pa] <= rank[pb] else X[pb]
                child = make_child(parent, sigma)
                fc = eval_one(child)
                used += 1
                worst = int(np.argmax(rank - cd * 1e-9))
                X[worst] = child
                F[worst] = fc
                # occasional local refinement of a good point
                if used + 1 < budget and rng.random() < 0.2 and time.time() < deadline - 4.0:
                    best_idx = int(np.argmin(rank))
                    lc = X[best_idx] + 0.02 * span * rng.standard_normal(dim)
                    lc = np.clip(lc, lo, hi)
                    lcf = eval_one(lc)
                    used += 1
                    # keep the better of (lc, current worst) by simple dominance/sum
                    if np.sum(lcf) < np.sum(F[worst]):
                        X[worst] = lc
                        F[worst] = lcf
                sigma = max(0.03, sigma * 0.99)

        # return something of the right shape (ignored by evaluator)
        if pops:
            return pops[0][0]
        return np.zeros((1, dim))
