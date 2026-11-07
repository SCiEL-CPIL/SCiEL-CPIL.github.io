# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG7 (3 objectives, 12 dims)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0          # safety margin under the 300 s wall-clock limit
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        # Vectorized wrapper: same number of problem() calls, but faster loop.
        vprob = np.vectorize(problem, otypes=[float] * n_obj)

        # Population size: 40 -> ~5 generations, leaves budget for exploration.
        pop = int(min(40, max(10, budget // 5)))

        X = lo + (hi - lo) * rng.random((pop, dim))
        F = vprob(X)
        used = pop

        def fast_non_dominated_sort(F):
            n = F.shape[0]
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            front = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        nd[p] += 1
                if nd[p] == 0:
                    front.append(p)
            fronts.append(front)
            i = 0
            while fronts[i]:
                nxt = []
                for p in fronts[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            nxt.append(q)
                i += 1
                if nxt:
                    fronts.append(nxt)
            return fronts

        def crowding(F, front):
            n = len(front)
            cd = np.zeros(n, dtype=float)
            if n <= 2:
                return cd
            for m in range(n_obj):
                vals = F[front, m]
                vmin, vmax = vals.min(), vals.max()
                rng_m = vmax - vmin
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                if rng_m > 0:
                    for k in range(1, n - 1):
                        cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / rng_m
            return cd

        while used < budget:
            if time.time() > deadline:
                break
            fronts = fast_non_dominated_sort(F)
            # Tournament selection (size 3) over (front, crowding) ranking.
            rank = np.zeros(F.shape[0], dtype=int)
            for fr, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = fr
            parents = []
            while len(parents) < pop:
                a = rng.integers(F.shape[0])
                b = rng.integers(F.shape[0])
                parents.append(a if rank[a] < rank[b] else b)
            parents = np.array(parents)
            X1 = X[parents]
            X2 = X[rng.integers(F.shape[0], size=pop)]
            mask = rng.random((pop, dim)) < 0.5
            child = np.where(mask, X1, X2)
            # Adaptive mutation: broad early, fine later.
            frac = used / float(budget)
            sigma = (0.25 * (1.0 - frac) + 0.05) * (hi - lo)
            child = child + sigma * rng.standard_normal((pop, dim))
            child = np.clip(child, lo, hi)
            fc = vprob(child)
            used += pop
            Xn = np.vstack([X, child])
            Fn = np.vstack([F, fc])
            fronts = fast_non_dominated_sort(Fn)
            Xsel, Fsel = [], []
            for front in fronts:
                if len(Xsel) + len(front) <= pop:
                    for idx in front:
                        Xsel.append(Xn[idx]); Fsel.append(Fn[idx])
                else:
                    cd = crowding(Fn, front)
                    need = pop - len(Xsel)
                    order = np.argsort(-cd)
                    for k in range(need):
                        idx = front[order[k]]
                        Xsel.append(Xn[idx]); Fsel.append(Fn[idx])
                    break
            X = np.array(Xsel)
            F = np.array(Fsel)
        return X
# EVOLVE-BLOCK-END
