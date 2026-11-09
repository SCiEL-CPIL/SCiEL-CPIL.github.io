"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Vectorized NSGA-II-style multi-objective search for RE37 (4 vars, 3 obj)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _eval(self, X, problem):
        """Evaluate a (N, dim) array, returning (N, n_obj)."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        F = np.empty((X.shape[0], self.n_obj), dtype=float)
        for i in range(X.shape[0]):
            F[i] = problem(X[i])
        return F

    def _non_dominated_sort(self, F):
        """Return (fronts, rank) where fronts is a list of index lists."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        rank = np.zeros(n, dtype=int)
        fronts = []
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1
            if nd[p] == 0:
                rank[p] = 0
                fronts.append([p])
        i = 0
        while i < len(fronts) and fronts[i]:
            nxt = []
            for p in fronts[i]:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        rank[q] = i + 1
                        nxt.append(q)
            i += 1
            if nxt:
                fronts.append(nxt)
        return fronts, rank

    def _crowding(self, F, idx):
        """Crowding distance for indices in a single front."""
        k = len(idx)
        cd = np.zeros(k)
        if k <= 2:
            return cd
        order = {v: j for j, v in enumerate(idx)}
        for m in range(self.n_obj):
            vals = np.array([F[idx[j], m] for j in range(k)])
            srt = np.argsort(vals, kind="stable")
            cd[srt[0]] = np.inf
            cd[srt[-1]] = np.inf
            rng = vals[srt[-1]] - vals[srt[0]]
            if rng > 0:
                for j in range(1, k - 1):
                    cd[srt[j]] += (vals[srt[j + 1]] - vals[srt[j - 1]]) / rng
        return cd

    def _select(self, F, k):
        """Select k indices via non-dominated sorting + crowding distance."""
        n = len(F)
        fronts, rank = self._non_dominated_sort(F)
        selected = []
        for fr in fronts:
            if len(selected) + len(fr) <= k:
                selected.extend(fr)
            else:
                need = k - len(selected)
                cd = self._crowding(F, fr)
                order = np.argsort(cd, kind="stable")
                selected.extend([fr[j] for j in order[need - 1:]])
                break
        return selected

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim, budget = self.dim, self.budget

        pop = int(min(24, max(8, budget // 8)))
        X = lo + span * rng.random((pop, dim))
        F = self._eval(X, problem)
        used = pop

        while used < budget:
            if time.time() > deadline - 5.0:
                break
            # Adaptive mutation scale (shrinks as budget is consumed).
            scale = 0.25 * max(0.15, 1.0 - used / float(budget))
            # Vectorized mutation of the whole population (one child per parent).
            Xc = X + scale * span * rng.standard_normal((pop, dim))
            Xc = np.clip(Xc, lo, hi)
            # Elitist union: keep best (pop) of parents + children.
            Xall = np.vstack([X, Xc])
            Fall = np.vstack([F, self._eval(Xc, problem)])
            used += pop
            keep = self._select(Fall, pop)
            X = Xall[keep]
            F = Fall[keep]
        return X
# EVOLVE-BLOCK-END
