"""Combined optimizer: Sobol lattice init + NSGA-II with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer using Sobol lattice initialization and NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        # Use Sobol sequence for deterministic, space-filling initialization
        pop_size = min(20, budget)
        if pop_size < 2:
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X

        # Generate Sobol points
        sobol = qmc.Sobol(d=dim, scramble=True, seed=0)
        # Use a power-of-2 sample size >= pop_size for Sobol
        sobol_n = 1
        while sobol_n < pop_size:
            sobol_n *= 2
        sobol_pts = sobol.random(sobol_n)
        # Scale to bounds
        X = lo + (hi - lo) * sobol_pts[:pop_size]
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # NSGA-II Loop
        while used + pop_size <= budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            ranks, fronts = self._non_dominated_sort(F)

            # Crowding distance
            cd = self._crowding_distance(F, fronts)

            # Create offspring via SBX crossover + polynomial mutation
            offspring_X = np.empty((pop_size, dim))
            n_pairs = pop_size // 2

            for i in range(n_pairs):
                p1_idx = self._tournament_select(ranks, cd, 2)
                p2_idx = self._tournament_select(ranks, cd, 2)

                x1, x2 = X[p1_idx], X[p2_idx]
                c1, c2 = self._sbx_crossover(x1, x2, lo, hi, 0.9, 20)
                c1 = self._polynomial_mutation(c1, lo, hi, 1.0 / dim, 20)
                c2 = self._polynomial_mutation(c2, lo, hi, 1.0 / dim, 20)

                offspring_X[i * 2] = c1
                offspring_X[i * 2 + 1] = c2

            if pop_size % 2 == 1:
                p1_idx = self._tournament_select(ranks, cd, 2)
                x1 = X[p1_idx]
                c1 = x1.copy()
                c1 = self._polynomial_mutation(c1, lo, hi, 1.0 / dim, 20)
                offspring_X[-1] = c1

            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring_X], dtype=np.float64)
            used += pop_size

            # Combine parent and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])

            # Non-dominated sort on combined population
            ranks_all, fronts_all = self._non_dominated_sort(F_all)
            cd_all = self._crowding_distance(F_all, fronts_all)

            # Select next population
            sort_keys = np.lexsort((-cd_all, ranks_all))
            next_idx = sort_keys[:pop_size]

            X = X_all[next_idx]
            F = F_all[next_idx]

        # Use remaining budget for local refinement of non-dominated points
        remaining = budget - used
        if remaining > 0:
            # Find non-dominated points
            ranks_final, fronts_final = self._non_dominated_sort(F)
            non_dom_idx = np.where(ranks_final == 0)[0]
            if len(non_dom_idx) > 0:
                # Distribute remaining evaluations among non-dominated points
                per_point = max(1, remaining // len(non_dom_idx))
                for idx in non_dom_idx:
                    if used >= budget:
                        break
                    # Local Nelder-Mead-like refinement using a scalarized objective
                    # Use a weighted sum with random weights to explore different parts of the front
                    for _ in range(min(per_point, budget - used)):
                        if time.time() > deadline:
                            break
                        weights = np.random.dirichlet(np.ones(n_obj))
                        base = X[idx]
                        # Small perturbation
                        step = 0.05 * (hi - lo)
                        candidate = base + step * np.random.randn(dim)
                        candidate = np.clip(candidate, lo, hi)
                        fc = problem(candidate)
                        used += 1
                        # Replace if it improves the scalarized objective or is non-dominated
                        scalar_old = np.dot(weights, F[idx])
                        scalar_new = np.dot(weights, fc)
                        if scalar_new < scalar_old:
                            X[idx] = candidate
                            F[idx] = fc

        return X

    def _non_dominated_sort(self, F):
        n = F.shape[0]
        ranks = np.zeros(n, dtype=int)
        fronts = []

        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_dom[i] += 1

        front = []
        for i in range(n):
            if n_dom[i] == 0:
                ranks[i] = 0
                front.append(i)

        while front:
            fronts.append(front)
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        ranks[j] = len(fronts)
                        next_front.append(j)
            front = next_front

        return ranks, fronts

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _crowding_distance(self, F, fronts):
        n = F.shape[0]
        n_obj = F.shape[1]
        cd = np.zeros(n)

        for front in fronts:
            m = len(front)
            if m <= 2:
                for i in front:
                    cd[i] = np.inf
                continue

            for k in range(n_obj):
                sorted_idx = sorted(front, key=lambda idx: F[idx, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                if m > 2:
                    f_min = F[sorted_idx[0], k]
                    f_max = F[sorted_idx[-1], k]
                    f_range = f_max - f_min

                    if f_range < 1e-14:
                        continue

                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i + 1], k] - F[sorted_idx[i - 1], k]) / f_range

        return cd

    def _tournament_select(self, ranks, cd, k=2):
        n = len(ranks)
        candidates = np.random.choice(n, k, replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best] and cd[c] > cd[best]:
                best = c
        return best

    def _sbx_crossover(self, x1, x2, lo, hi, p_c=0.9, eta_c=20):
        if np.random.random() > p_c:
            return x1.copy(), x2.copy()

        c1, c2 = x1.copy(), x2.copy()
        for i in range(len(x1)):
            if np.random.random() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    x_min = min(x1[i], x2[i])
                    x_max = max(x1[i], x2[i])
                    diff = x_max - x_min

                    u = np.random.random()
                    beta = 1.0 + 2.0 * (x1[i] - x_min) / diff
                    alpha = 2.0 - beta ** (-eta_c - 1.0)
                    if u <= 1.0 / alpha:
                        beta_q = (u * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))

                    c1[i] = 0.5 * ((x1[i] + x2[i]) - beta_q * (x1[i] - x2[i]))

                    u = np.random.random()
                    beta = 1.0 + 2.0 * (x2[i] - x_min) / diff
                    alpha = 2.0 - beta ** (-eta_c - 1.0)
                    if u <= 1.0 / alpha:
                        beta_q = (u * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))

                    c2[i] = 0.5 * ((x1[i] + x2[i]) + beta_q * (x1[i] - x2[i]))

        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, eta_m=20, p_m=1.0 / 12):
        x = x.copy()
        for i in range(len(x)):
            if np.random.random() < p_m:
                y = x[i]
                y_min = lo[i]
                y_max = hi[i]
                diff = y_max - y_min
                if diff < 1e-14:
                    continue

                u = np.random.random()
                if u < 0.5:
                    xy = (y - y_min) / diff
                    beta = 1.0 + 2.0 * (1.0 - xy)
                    alpha = 2.0 + beta * (eta_m + 1.0)
                    delta = (2.0 * (u + 1.0 / alpha) ** (1.0 / (eta_m + 1.0)) - 1.0)
                else:
                    xy = (y_max - y) / diff
                    beta = 1.0 + 2.0 * xy
                    alpha = 2.0 + beta * (eta_m + 1.0)
                    delta = 1.0 - (2.0 * (1.0 - u + 1.0 / alpha)) ** (1.0 / (eta_m + 1.0))

                y = y + delta * diff
                x[i] = np.clip(y, y_min, y_max)
        return x
# EVOLVE-BLOCK-END
