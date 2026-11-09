"""Multi-objective optimizer on WFG4 (3 objectives, 12 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style evolutionary multi-objective search with a wall-clock guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        if np.any(span <= 0):
            span = np.where(span <= 0, 1.0, span)

        rng = np.random
        budget = int(self.budget)

        # Population size: keep small so each generation is cheap.
        pop = max(10, min(24, budget // 8))

        def rand_pop(n):
            return lo + span * rng.random((n, dim))

        X = rand_pop(pop)
        F = np.array([np.asarray(problem(x), dtype=float) for x in X])
        used = pop

        # Fast non-dominated sorting (returns list of fronts, each an index array).
        def fronts(f):
            n = len(f)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            front0 = []
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(f[p] <= f[q]) and np.any(f[p] < f[q]):
                        S[p].append(q)
                        nd[p] += 1
                    elif np.all(f[q] <= f[p]) and np.any(f[q] < f[p]):
                        nd[q] += 1
                if nd[p] == 0:
                    rank[p] = 0
                    front0.append(p)
            fronts_list = [np.array(front0, dtype=int)]
            i = 0
            while len(fronts_list[i]) > 0:
                nxt = []
                for p in fronts_list[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            rank[q] = i + 1
                            nxt.append(q)
                i += 1
                fronts_list.append(np.array(nxt, dtype=int))
            return fronts_list

        def crowding(f, idx):
            if len(idx) <= 2:
                return np.full(len(idx), 1e9)
            d = np.zeros(len(idx), dtype=float)
            m = len(idx)
            for k in range(n_obj):
                vals = f[idx, k]
                order = np.argsort(vals, kind='stable')
                smin = vals[order[0]]
                smax = vals[order[-1]]
                rngv = smax - smin
                for pos in range(m):
                    gi = order[pos]
                    if pos == 0 or pos == m - 1:
                        d[gi] = 1e9
                    else:
                        if rngv > 1e-15:
                            d[gi] += (vals[order[pos + 1]] - vals[order[pos - 1]]) / rngv
                        else:
                            d[gi] += 0.0
            return d

        while used < budget:
            if time.time() > deadline:
                break

            # Build offspring the same size as the population.
            n_off = pop
            off = []
            for _ in range(n_off):
                if used + 1 > budget or time.time() > deadline:
                    break
                # Binary tournament on (rank, crowding).
                rank = np.zeros(len(F), dtype=int)
                fr = fronts(F)
                for r, fridx in enumerate(fr):
                    rank[fridx] = r
                cd = np.zeros(len(F), dtype=float)
                for fridx in fr:
                    cd[fridx] = crowding(F, fridx)
                a = int(rng.integers(len(F)))
                b = int(rng.integers(len(F)))
                if (rank[a], -cd[a]) < (rank[b], -cd[b]):
                    pa = a
                else:
                    pa = b
                c = int(rng.integers(len(F)))
                dd = int(rng.integers(len(F)))
                if (rank[c], -cd[c]) < (rank[dd], -cd[dd]):
                    pb = c
                else:
                    pb = dd

                # Simulated binary crossover + polynomial mutation.
                x1 = X[pa]
                x2 = X[pb]
                child = x1.copy()
                # Blend with SBX-like mixing.
                eta = 20.0
                for j in range(dim):
                    if rng.random() < 0.5:
                        # keep x1
                        continue
                    # blend toward x2
                    child[j] = 0.5 * (x1[j] + x2[j])
                # mutation
                for j in range(dim):
                    if rng.random() < 0.15:
                        child[j] += 0.1 * span[j] * rng.standard_normal()
                child = np.clip(child, lo, hi)
                fc = np.asarray(problem(child), dtype=float)
                used += 1
                off.append((child, fc))

            if not off:
                break

            offX = np.array([o[0] for o in off])
            offF = np.array([o[1] for o in off])

            # Combine and select.
            CX = np.vstack([X, offX])
            CF = np.vstack([F, offF])
            fr = fronts(CF)
            rank = np.zeros(len(CF), dtype=int)
            for r, fridx in enumerate(fr):
                rank[fridx] = r
            cd = np.zeros(len(CF), dtype=float)
            for fridx in fr:
                cd[fridx] = crowding(CF, fridx)

            # Select first pop by (rank, -crowding).
            order = np.lexsort((-cd, rank))
            keep = order[:pop]
            X = CX[keep]
            F = CF[keep]

        return X
# EVOLVE-BLOCK-END
