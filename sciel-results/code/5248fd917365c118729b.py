"""NSGA-II on WFG7 with time guard + local polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        C = np.zeros(n, dtype=int)
        fronts = [[]]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    C[i] += 1
            if C[i] == 0:
                fronts[0].append(i)
        k = 0
        while fronts[k]:
            nxt = []
            for i in fronts[k]:
                for j in S[i]:
                    C[j] -= 1
                    if C[j] == 0:
                        nxt.append(j)
            k += 1
            if nxt:
                fronts.append(nxt)
        rank = np.zeros(n, dtype=int)
        for fi in range(len(fronts)):
            for i in fronts[fi]:
                rank[i] = fi
        return rank

    def _crowding(self, F):
        n = len(F)
        cd = np.zeros(n)
        if n <= 2:
            return cd
        for m in range(F.shape[1]):
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            rng = F[order[-1], m] - F[order[0], m]
            if rng > 0:
                for i in range(1, n - 1):
                    cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / rng
        return cd

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        pop = min(20, max(6, self.budget // 10))
        X = lo + (hi - lo) * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop
        while used < self.budget and time.time() < deadline:
            rank = self._non_dominated_sort(F)
            cd = self._crowding(F)
            idx = np.lexsort((cd, rank))
            keep = pop // 2
            parents = X[idx[:keep]]
            c1 = parents[rng.integers(len(parents))]
            c2 = parents[rng.integers(len(parents))]
            child = 0.5 * (c1 + c2) + 0.15 * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            worst = idx[-1]
            X[worst], F[worst] = child, fc
        # local polish
        if used < self.budget and time.time() < deadline:
            for i in range(len(X)):
                if time.time() > deadline or used >= self.budget:
                    break
                best_x, best_f = X[i].copy(), F[i].copy()
                step = 0.05 * (hi - lo)
                for _ in range(6):
                    if time.time() > deadline or used >= self.budget:
                        break
                    for d in range(self.dim):
                        if time.time() > deadline or used >= self.budget:
                            break
                        for sgn in (1.0, -1.0):
                            if time.time() > deadline or used >= self.budget:
                                break
                            cand = best_x.copy()
                            cand[d] += sgn * step
                            cand = np.clip(cand, lo, hi)
                            cf = np.asarray(problem(cand), dtype=float)
                            used += 1
                            if np.all(cf <= best_f) and np.any(cf < best_f):
                                best_x, best_f = cand, cf
                    step *= 0.5
                X[i], F[i] = best_x, best_f
        return X
# EVOLVE-BLOCK-END
