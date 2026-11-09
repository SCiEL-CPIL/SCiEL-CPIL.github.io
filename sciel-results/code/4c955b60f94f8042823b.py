"""NSGA-II style multi-objective optimizer for DTLZ2."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II multi-objective evolutionary optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: keep it moderate to fit budget
        N = min(30, max(8, self.budget // 6))

        # Initial population
        X = lo + span * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        # SBX parameters
        eta_c = 20.0
        eta_m = 20.0
        cx_prob = 0.9
        mut_rate = 1.0 / self.dim

        while used + 2 * N <= self.budget:
            # Generate offspring
            offspring_X = []
            offspring_F = []
            for _ in range(N):
                # Tournament select two parents
                p1 = self._tournament_select(F, N_t=2, rng=rng)
                p2 = self._tournament_select(F, N_t=2, rng=rng)

                # Crossover
                if rng.random() < cx_prob:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutation
                c1 = self._poly_mutation(c1, lo, hi, mut_rate, eta_m, rng)
                c2 = self._poly_mutation(c2, lo, hi, mut_rate, eta_m, rng)

                # Evaluate
                f1 = problem(c1)
                used += 1
                f2 = problem(c2)
                used += 1

                offspring_X.extend([c1, c2])
                offspring_F.extend([f1, f2])

            # Combine
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])

            # Non-dominated sorting
            fronts = self._non_dominated_sort(combined_F)

            # Crowding distance per front
            crowding = np.zeros(len(combined_F))
            for front in fronts:
                if len(front) > 2:
                    cd = self._crowding_distance(combined_F[front])
                    crowding[front] = cd
                elif len(front) == 2:
                    crowding[front] = np.inf
                else:
                    crowding[front] = np.inf

            # Select next generation
            next_idx = []
            for front in fronts:
                if len(next_idx) + len(front) <= N:
                    next_idx.extend(front)
                else:
                    front_sorted = sorted(front, key=lambda i: -crowding[i])
                    needed = N - len(next_idx)
                    next_idx.extend(front_sorted[:needed])
                    break

            X = combined_X[next_idx]
            F = combined_F[next_idx]

        return X

    def _tournament_select(self, F, N_t=2, rng=None):
        """Binary tournament selection using non-dominated rank and crowding distance."""
        n = len(F)
        # Compute ranks and crowding for tournament
        fronts = self._non_dominated_sort(F)
        ranks = np.zeros(n, dtype=int)
        for rank, front in enumerate(fronts):
            ranks[front] = rank

        crowding = np.zeros(n)
        for front in fronts:
            if len(front) > 2:
                cd = self._crowding_distance(F[front])
                crowding[front] = cd
            else:
                crowding[front] = np.inf

        # Binary tournament
        i = rng.integers(n)
        j = rng.integers(n)
        if ranks[i] < ranks[j]:
            return F[i]
        elif ranks[j] < ranks[i]:
            return F[j]
        else:
            if crowding[i] > crowding[j]:
                return F[i]
            else:
                return F[j]

    def _sbx_crossover(self, p1, p2, lo, hi, eta, rng):
        """Simulated Binary Crossover."""
        c1, c2 = p1.copy(), p2.copy()
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if p1[i] < p2[i]:
                        x1, x2 = p1[i], p2[i]
                    else:
                        x1, x2 = p2[i], p1[i]
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    c1[i] = 0.5 * ((x1 + x2) - beta * (x2 - x1))
                    c2[i] = 0.5 * ((x1 + x2) + beta * (x2 - x1))
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def _poly_mutation(self, x, lo, hi, mut_rate, eta, rng):
        """Polynomial mutation."""
        x = x.copy()
        for i in range(len(x)):
            if rng.random() < mut_rate:
                y = x[i]
                yl, yu = lo[i], hi[i]
                if yu - yl < 1e-14:
                    continue
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                u = rng.random()
                if u < 0.5:
                    xy = (2 * u + (1 - 2 * u) * (delta1 ** (eta + 1))) ** (1.0 / (eta + 1))
                    mut_pow = 1.0 + (eta + 1) * (1 - xy)
                else:
                    xy = (2 * (1 - u) + 2 * (u - 0.5) * (delta2 ** (eta + 1))) ** (1.0 / (eta + 1))
                    mut_pow = 1.0 + (eta + 1) * (xy - 1)
                y = y + (yu - yl) * (mut_pow - 1)
                x[i] = np.clip(y, lo[i], hi[i])
        return x

    def _non_dominated_sort(self, F):
        """Non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]
        n_doms = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_doms[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_doms[i] += 1

        fronts = []
        front = [i for i in range(n) if n_doms[i] == 0]
        if front:
            fronts.append(front)
        k = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_doms[j] -= 1
                    if n_doms[j] == 0:
                        next_front.append(j)
            k += 1
            front = next_front
            if front:
                fronts.append(front)

        return fronts

    def _crowding_distance(self, F):
        """Compute crowding distance for a set of points in one front."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)

        m = F.shape[1]
        cd = np.zeros(n)
        for k in range(m):
            order = np.argsort(F[:, k])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            f_min = F[order[0], k]
            f_max = F[order[-1], k]
            if f_max - f_min < 1e-14:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (F[order[i + 1], k] - F[order[i - 1], k]) / (f_max - f_min)
        return cd
# EVOLVE-BLOCK-END
