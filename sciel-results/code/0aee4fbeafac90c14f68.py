"""Multi-objective optimizer on WFG9: NSGA-II style with global RNG + deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective search (NSGA-II style) minimizing all objectives."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _nondominated_idx(F):
        """Indices of non-dominated rows in F (minimization)."""
        n = F.shape[0]
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            other = F != F[i]          # strictly better in some objective
            better = np.any(other, axis=1)
            if np.any(better):
                keep[i] = False
        return np.where(keep)[0]

    @staticmethod
    def _crowding(F):
        """Crowding distance per row (standard NSGA-II)."""
        n = F.shape[0]
        dist = np.zeros(n)
        for m in range(F.shape[1]):
            order = np.argsort(F[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            span = F[order[-1], m] - F[order[0], m]
            if span > 0:
                for k in range(1, n - 1):
                    dist[order[k]] += (F[order[k + 1], m] - F[order[k - 1], m]) / span
        return dist

    def _select_parents(self, F, rng):
        """Binary tournament on (rank, crowding)."""
        n = F.shape[0]
        # ranks via iterative non-dominated layers
        remaining = list(range(n))
        rank = np.zeros(n, dtype=int)
        r = 0
        while remaining:
            idx = np.array(remaining)
            nd = self._nondominated_idx(F[idx])
            layer = idx[nd]
            for j in layer:
                rank[j] = r
            remaining = [j for j in remaining if j not in set(layer.tolist())]
            r += 1
        crowd = self._crowding(F)

        def key(i):
            return (rank[i], -crowd[i])

        def pick():
            a = rng.integers(n)
            b = rng.integers(n)
            if key(a) <= key(b):
                return a
            return b

        return pick(), pick()

    def __call__(self, problem, bounds):
        dim, n_obj, budget = self.dim, self.n_obj, self.budget
        rng = np.random  # use the evaluator's global seed (seed 0 / seed 1)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        deadline = time.time() + 280.0   # safety margin vs 300s limit

        pop = int(min(40, max(10, budget // 6)))
        X = lo + span * rng.random_sample((pop, dim))
        F = np.empty((pop, n_obj))
        used = 0
        for i in range(pop):
            F[i] = problem(X[i])
            used += 1

        def make_child():
            a, b = self._select_parents(F, rng)
            if rng.random_sample() < 0.8:
                # SBX crossover
                alpha = 20.0
                child = np.empty(dim)
                for d in range(dim):
                    x1, x2 = X[a, d], X[b, d]
                    if x1 == x2 or rng.random_sample() > 0.5:
                        child[d] = x1 if rng.random_sample() < 0.5 else x2
                        continue
                    u = rng.random_sample()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (alpha + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (alpha + 1))
                    c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                    child[d] = c1 if rng.random_sample() < 0.5 else c2
            else:
                child = X[a].copy()
            # polynomial mutation
            eta = 20.0
            for d in range(dim):
                if rng.random_sample() < 0.9:
                    u = rng.random_sample()
                    if u <= 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    child[d] += delta_q * span[d]
            return np.clip(child, lo, hi)

        while used < budget:
            if time.time() > deadline:
                break
            child = make_child()
            fc = problem(child)
            used += 1
            # replace worst (max rank, then max crowding)
            n = F.shape[0]
            remaining = list(range(n))
            rank = np.zeros(n, dtype=int)
            r = 0
            while remaining:
                idx = np.array(remaining)
                nd = self._nondominated_idx(F[idx])
                layer = idx[nd].tolist()
                for j in layer:
                    rank[j] = r
                remaining = [j for j in remaining if j not in layer]
                r += 1
            crowd = self._crowding(F)
            worst = int(np.argmax(rank + 1e-9 * crowd))
            X[worst] = child
            F[worst] = fc
        return X
# EVOLVE-BLOCK-END
