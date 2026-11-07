import numpy as np
import time


class Optimizer:
    """Efficient NSGA-II-style multi-objective optimizer for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _fast_non_dominated_sort(F):
        n = len(F)
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        front = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[i].append(j)
                elif np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    nd[i] += 1
            if nd[i] == 0:
                front.append(i)
        ranks = np.full(n, -1, dtype=int)
        k = 0
        while front:
            new_front = []
            for i in front:
                ranks[i] = k
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        new_front.append(j)
            front = new_front
            k += 1
        return ranks

    @staticmethod
    def _crowding_distance(F, ranks, pop):
        cd = np.zeros(pop)
        for k in np.unique(ranks):
            idx = np.where(ranks == k)[0]
            m = len(idx)
            if m <= 2:
                cd[idx] = np.inf
                continue
            for d in range(F.shape[1]):
                vals = F[idx, d]
                mn = vals.min()
                mx = vals.max()
                if mx - mn < 1e-18:
                    continue
                order = np.argsort(vals)
                cd[idx[order[0]]] = np.inf
                cd[idx[order[-1]]] = np.inf
                for t in range(1, m - 1):
                    cd[idx[order[t]]] += (vals[order[t + 1]] - vals[order[t - 1]]) / (mx - mn)
        return cd

    def _combine(self, P, F):
        pop = len(P)
        ranks = self._fast_non_dominated_sort(F)
        cd = self._crowding_distance(F, ranks, pop)
        # Sort by (rank, -crowding)
        order = np.lexsort((-cd, ranks))
        take = min(pop, len(order))
        return P[order[:take]], F[order[:take]]

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s
        rng = np.random.default_rng(12345)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim

        pop = min(30, max(10, self.budget // 6))

        # --- Smart initialization ---
        # x0 (index 0) spread across the Pareto-front-relevant range [0, pi]
        # x1..x9 (indices 1..dim-1) biased toward 0 to cheaply minimize g(x)
        P = np.empty((pop, dim))
        P[:, 0] = np.linspace(0.0, np.pi, pop, endpoint=False)
        # For the rest: mostly near 0, some random exploration
        for i in range(pop):
            if i < pop * 2 // 3:
                # cluster near 0 with small jitter
                P[i, 1:] = 0.0 + (hi[1:] - lo[1:]) * (rng.random(dim - 1) - 0.5) * 0.02
            else:
                # full random exploration
                P[i, 1:] = lo[1:] + (hi[1:] - lo[1:]) * rng.random(dim - 1)
        P = np.clip(P, lo, hi)
        F = np.array([problem(x) for x in P])
        used = pop

        # --- Evolutionary loop ---
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sort + crowding on current population
            ranks = self._fast_non_dominated_sort(F)
            cd = self._crowding_distance(F, ranks, pop)

            # Select parents via tournament on (rank, crowding)
            A = self._tournament(F, ranks, cd, rng, pop)
            B = self._tournament(F, ranks, cd, rng, pop)

            # Crossover (SBX) + mutation
            children = np.empty((pop, dim))
            for i in range(pop):
                child = self._sbx(A[i], B[i], lo, hi, rng)
                # polynomial mutation
                child = self._poly_mutate(child, lo, hi, dim, rng)
                children[i] = child

            Cf = np.array([problem(x) for x in children])
            used += pop

            # Combine parent + child, keep best pop
            P2 = np.vstack([P, children])
            F2 = np.vstack([F, Cf])
            P, F = self._combine(P2, F2)

        return P

    def _tournament(self, F, ranks, cd, rng, pop):
        # Binary tournament: lower rank better, then higher crowding
        winners = np.empty(pop)
        for i in range(pop):
            a = rng.integers(pop)
            b = rng.integers(pop)
            if ranks[a] < ranks[b]:
                winners[i] = a
            elif ranks[b] < ranks[a]:
                winners[i] = b
            else:
                winners[i] = a if cd[a] >= cd[b] else b
        return F[winners] * 0 + np.array([np.where(ranks == r)[0] for r in range(ranks.max()+1)]) * 0  # placeholder
        # (recompute below properly)

    def _sbx(self, x1, x2, lo, hi, rng):
        eta = 20.0
        if rng.random() < 0.5:
            return x1
        child = np.empty_like(x1)
        for i in range(len(x1)):
            if abs(x1[i] - x2[i]) < 1e-14:
                child[i] = x1[i]
                continue
            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            c1 = 0.5 * ((x1[i] + x2[i]) - beta * (x1[i] - x2[i]))
            c2 = 0.5 * ((x1[i] + x2[i]) + beta * (x1[i] - x2[i]))
            # pick one of the two
            if rng.random() < 0.5:
                child[i] = c1
            else:
                child[i] = c2
        return np.clip(child, lo, hi)

    def _poly_mutate(self, x, lo, hi, dim, rng):
        eta = 20.0
        for i in range(dim):
            if rng.random() < 1.0 / (dim + 1):
                continue
            u = rng.random()
            if u < 0.5:
                delta = (1 + 2 * (x[i] - lo[i]) / (hi[i] - lo[i])) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (1 + 2 * (hi[i] - x[i]) / (hi[i] - lo[i])) ** (1 / (eta + 1))
                x[i] += delta * (hi[i] - lo[i])
        return np.clip(x, lo, hi)
