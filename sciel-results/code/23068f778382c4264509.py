"""Multi-objective optimizer for WFG7 (3 objectives, 12 vars).

Strategy: NSGA-II with front-aware initialization exploiting WFG7's known
spherical Pareto front geometry.
"""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = self.n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        mid = 0.5 * (lo + hi)

        rng = np.random.default_rng(42)
        used = 0

        def evaluate(x):
            nonlocal used
            used += 1
            return np.asarray(problem(x), dtype=float)

        pop_size = 30
        n_front_seeds = 12

        def front_seed():
            v = rng.random(n_obj) + 0.1
            v /= np.linalg.norm(v)
            x = mid.copy()
            for k in range(min(n_obj, dim)):
                x[k] = lo[k] + v[k] * span[k] * 0.8 + span[k] * 0.1
            for k in range(n_obj, dim):
                x[k] = lo[k] + span[k] * rng.random()
            return np.clip(x, lo, hi)

        X = np.empty((pop_size, dim))
        for i in range(n_front_seeds):
            X[i] = front_seed()
        for i in range(n_front_seeds, pop_size):
            X[i] = lo + span * rng.random(dim)

        F = np.empty((pop_size, n_obj))
        for i in range(pop_size):
            F[i] = evaluate(X[i])

        def non_dominated_sort(F):
            n = len(F)
            dom = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dom[i, j] = True
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dom[j, i] = True
            fronts = []
            remaining = list(range(n))
            while remaining:
                front = [i for i in remaining
                         if not any(dom[j, i] for j in remaining if j != i)]
                if not front:
                    front = remaining[:]
                fronts.append(front)
                remaining = [i for i in remaining if i not in front]
            ranks = np.zeros(n, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r
            return fronts, ranks

        def crowding(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                mn, mx = vals.min(), vals.max()
                if mx - mn < 1e-12:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (mx - mn)
            return cd

        while used < budget and time.time() < deadline:
            fronts, ranks = non_dominated_sort(F)
            crowd = np.zeros(len(F))
            for front in fronts:
                cd = crowding(F, np.array(front))
                for k, idx in enumerate(front):
                    crowd[idx] = cd[k]

            n_off = min(pop_size, budget - used)
            if n_off <= 0:
                break

            offX = np.empty((n_off, dim))
            offF = np.empty((n_off, n_obj))
            progress = used / max(1, budget)
            mut_scale = 0.1 * (1.0 - 0.4 * progress)

            for c in range(n_off):
                if used >= budget or time.time() > deadline:
                    break
                def select():
                    idxs = rng.integers(0, len(F), size=3)
                    best = idxs[0]
                    for idx in idxs[1:]:
                        if ranks[idx] < ranks[best] or (ranks[idx] == ranks[best] and crowd[idx] > crowd[best]):
                            best = idx
                    return best
                p1 = select()
                p2 = select()
                child = X[p1].copy()
                if rng.random() < 0.9:
                    a, b = sorted(rng.choice(dim, size=2, replace=False))
                    child[a:b] = X[p2][a:b]
                child = child + mut_scale * span * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                offX[c] = child
                offF[c] = evaluate(child)

            popX = np.vstack([X, offX])
            popF = np.vstack([F, offF])
            fronts, ranks = non_dominated_sort(popF)
            crowd = np.zeros(len(popF))
            for front in fronts:
                cd = crowding(popF, np.array(front))
                for k, idx in enumerate(front):
                    crowd[idx] = cd[k]
            order = np.lexsort((-crowd, ranks))
            keep = order[:pop_size]
            X = popX[keep]
            F = popF[keep]

        return X
