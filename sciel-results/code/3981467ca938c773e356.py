"""NSGA-II multi-objective optimizer for DTLZ4."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II: fast non-dominated sorting, crowding distance, SBX crossover, polynomial mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random

        # Population size and generations
        pop_size = 20
        n_gen = self.budget // pop_size - 1
        if n_gen < 1:
            n_gen = 1

        # Initialize population: mix of random and origin-biased to exploit DTLZ4 bias
        X = np.empty((pop_size, dim))
        # Half uniform random, half biased toward 0 (DTLZ4 has x_i^beta bias)
        for i in range(pop_size):
            if i < pop_size // 2:
                X[i] = lo + (hi - lo) * rng.random(dim)
            else:
                # Bias toward 0: use beta distribution with alpha<1
                X[i] = lo + (hi - lo) * rng.beta(0.5, 5.0, size=dim)

        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def non_dominated_sort(F):
            """Return list of fronts (each front is list of indices)."""
            n = len(F)
            S = [set() for _ in range(n)]  # S[i]: set of indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of individuals dominating i
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            front0 = [i for i in range(n) if n_dom[i] == 0]
            fronts.append(front0)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    k += 1
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for individuals in a front."""
            n = len(front_indices)
            if n == 0:
                return {}
            dist = np.zeros(n)
            for m in range(n_obj):
                sorted_idx = np.argsort(F[front_indices, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_min = F[front_indices, m][sorted_idx[0]]
                f_max = F[front_indices, m][sorted_idx[-1]]
                if f_max - f_min > 1e-12:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (
                            F[front_indices, m][sorted_idx[k + 1]] -
                            F[front_indices, m][sorted_idx[k - 1]]
                        ) / (f_max - f_min)
            return {front_indices[i]: dist[i] for i in range(n)}

        def sbx_crossover(x1, x2):
            """Simulated binary crossover."""
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            eta = 20.0
            c1 = x1.copy()
            c2 = x2.copy()
            for i in range(dim):
                if x1[i] == x2[i]:
                    continue
                if rng.random() < 0.5:
                    continue
                beta = 1.0 + 2.0 * min(x1[i], x2[i]) / abs(x1[i] - x2[i] + 1e-12)
                alpha1 = (beta + 1) / (beta + 2)
                alpha2 = (beta + 1) / (beta + 2)
                r = rng.random()
                if r <= alpha1:
                    betaq = (r / alpha1) ** (1.0 / (eta + 1.0))
                else:
                    betaq = ((1.0 - r) / (1.0 - alpha1)) ** (1.0 / (eta + 1.0))
                c1[i] = 0.5 * ((1 + betaq) * x1[i] + (1 - betaq) * x2[i])
                r2 = rng.random()
                if r2 <= alpha2:
                    betaq2 = (r2 / alpha2) ** (1.0 / (eta + 1.0))
                else:
                    betaq2 = ((1.0 - r2) / (1.0 - alpha2)) ** (1.0 / (eta + 1.0))
                c2[i] = 0.5 * ((1 + betaq2) * x2[i] + (1 - betaq2) * x1[i])
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            return c1, c2

        def polynomial_mutate(x):
            """Polynomial mutation."""
            eta_m = 20.0
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < 0.8:  # mutation rate
                    delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i] + 1e-12)
                    delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i] + 1e-12)
                    r = rng.random()
                    if r < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * r + (1.0 - 2.0 * r) * xy ** (eta_m + 1.0)
                        deltaq = val ** (1.0 / (eta_m + 1.0)) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * xy ** (eta_m + 1.0)
                        deltaq = 1.0 - val ** (1.0 / (eta_m + 1.0))
                    x_mut[i] += (hi[i] - lo[i]) * deltaq
            return np.clip(x_mut, lo, hi)

        # NSGA-II main loop
        for gen in range(n_gen):
            # Create offspring
            offspring_X = np.empty((pop_size, dim))
            for i in range(pop_size):
                # Binary tournament selection
                def tournament():
                    idx1 = rng.randint(pop_size)
                    idx2 = rng.randint(pop_size)
                    # Get ranks and distances
                    fronts = non_dominated_sort(F)
                    rank = {}
                    for rank_idx, front in enumerate(fronts):
                        for idx in front:
                            rank[idx] = rank_idx
                    # Crowding for front 0
                    cd = {}
                    for front in fronts:
                        cd.update(crowding_distance(F, front))
                    r1 = rank[idx1]
                    r2 = rank[idx2]
                    if r1 < r2:
                        return idx1
                    elif r2 < r1:
                        return idx2
                    else:
                        if cd.get(idx1, 0) > cd.get(idx2, 0):
                            return idx1
                        else:
                            return idx2

                p1 = tournament()
                p2 = tournament()
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                offspring_X[2 * i] = c1
                if 2 * i + 1 < pop_size:
                    offspring_X[2 * i + 1] = c2

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X], dtype=float)
            used += pop_size

            # Combine parent and offspring
            all_X = np.vstack([X, offspring_X])
            all_F = np.vstack([F, offspring_F])

            # Non-dominated sort and select
            fronts = non_dominated_sort(all_F)
            selected_idx = []
            for front in fronts:
                if len(selected_idx) + len(front) <= pop_size:
                    selected_idx.extend(front)
                else:
                    # Need to trim this front by crowding distance
                    cd = crowding_distance(all_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    selected_idx.extend(sorted_front[:pop_size - len(selected_idx)])
                    break

            X = all_X[selected_idx]
            F = all_F[selected_idx]

        return X

# EVOLVE-BLOCK-END
