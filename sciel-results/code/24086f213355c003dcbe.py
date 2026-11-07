"""Multi-objective optimizer on WFG9 using NSGA-II."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II multi-objective optimizer with a wall-clock deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _fast_non_dominated_sort(self, F):
        """Return (fronts, domination_count, dominated_indices) for F (N x n_obj)."""
        n = F.shape[0]
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []
        current = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[j] += 1
            if n_dom[i] == 0:
                current.append(i)
                F[i, -1] = 0
        fronts.append(current)
        k = 0
        while current:
            nxt = []
            for i in current:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        nxt.append(j)
                        F[j, -1] = k + 1
            k += 1
            current = nxt
            if current:
                fronts.append(current)
        return fronts, n_dom, S

    def _crowding_distance(self, F, indices):
        """Crowding distance for the given indices (subset rows of F)."""
        d = np.zeros(len(indices))
        if len(indices) <= 2:
            return d
        m = F.shape[1] - 1  # exclude rank column
        for k in range(m):
            vals = F[indices, k]
            order = np.argsort(vals)
            lo = vals[order[0]]
            hi = vals[order[-1]]
            rng = hi - lo
            if rng < 1e-12:
                continue
            d[order[0]] = np.inf
            d[order[-1]] = np.inf
            for i in range(1, len(order) - 1):
                d[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / rng
        return d

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 330.0 - 6.0  # safety margin before the 330s wall-clock kill
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        def eval_batch(X):
            # X: (M, dim). Call problem per row; each call counts once.
            return np.array([problem(x) for x in X], dtype=float)

        # --- population sizing ---
        pop = int(min(40, max(12, budget // 5)))
        # reserve a few evals for safety; generations = (budget - pop) // pop
        gen_budget = budget - pop
        n_gen = max(1, gen_budget // pop)

        # --- initial population ---
        rng = np.random.default_rng(0)
        X = lo + span * rng.random((pop, dim))
        F = eval_batch(X)
        used = pop
        recorded = [F.copy()]  # keep all evaluated objective vectors

        # rank column appended
        F_rank = np.hstack([F, np.zeros((pop, 1))])

        # --- NSGA-II main loop ---
        for g in range(n_gen):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            # non-dominated sorting
            F_work = F_rank.copy()
            fronts, n_dom, S = self._fast_non_dominated_sort(F_work)

            # crowding within each front
            cd = np.zeros(pop)
            for front in fronts:
                d = self._crowding_distance(F_work, front)
                for idx, dist in zip(front, d):
                    cd[idx] = dist

            # parent selection: tournament on (rank, crowding)
            def rank_of(i):
                return int(F_work[i, -1])

            def tournament():
                a = rng.integers(pop)
                b = rng.integers(pop)
                ra, rb = rank_of(a), rank_of(b)
                if ra != rb:
                    return a if ra < rb else b
                return a if cd[a] >= cd[b] else b

            # build offspring
            n_off = min(pop, budget - used)
            off_X = np.zeros((n_off, dim))
            for i in range(n_off):
                p1 = tournament()
                p2 = tournament()
                # crossover
                if rng.random() < 0.9:
                    alpha = rng.random(dim)
                    child = alpha * X[p1] + (1.0 - alpha) * X[p2]
                else:
                    child = X[p1].copy()
                # mutation
                if rng.random() < 0.9:
                    mask = rng.random(dim) < 0.2
                    child[mask] += 0.15 * span[mask] * rng.standard_normal(mask.sum())
                off_X[i] = np.clip(child, lo, hi)

            off_F = eval_batch(off_X)
            used += n_off
            recorded.append(off_F.copy())

            # combine parents + offspring, select next pop
            comb_X = np.vstack([X, off_X])
            comb_F = np.vstack([F, off_F])
            comb_F_rank = np.hstack([comb_F, np.zeros((2 * pop, 1))])
            cfronts, _, _ = self._fast_non_dominated_sort(comb_F_rank)

            next_idx = []
            for front in cfronts:
                if len(next_idx) + len(front) <= pop:
                    next_idx.extend(front)
                else:
                    need = pop - len(next_idx)
                    d = self._crowding_distance(comb_F_rank, front)
                    order = np.argsort(d)[::-1]
                    next_idx.extend([front[i] for i in order[:need]])
                    break
            next_idx = np.array(next_idx[:pop])
            X = comb_X[next_idx]
            F = comb_F[next_idx]
            F_rank = comb_F_rank[next_idx]

            if time.time() > deadline:
                break

        return X
# EVOLVE-BLOCK-END
