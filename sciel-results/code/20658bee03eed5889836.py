"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 3))
        if pop_size > budget:
            pop_size = max(10, budget // 2)

        # Initial population
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Returns list of fronts (list of indices)."""
            n = len(F)
            S = [set() for _ in range(n)]
            n_doms = [0] * n
            fronts = []
            front0 = set()

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_doms[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_doms[i] += 1

            for i in range(n):
                if n_doms[i] == 0:
                    front0.add(i)

            k = 0
            while front0:
                fronts.append(list(front0))
                next_front = set()
                for i in front0:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.add(j)
                front0 = next_front
                k += 1

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n == 0:
                return np.array([])
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            idx = list(indices)
            Fsub = F[idx]

            for m in range(n_obj):
                sorted_idx = np.argsort(Fsub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = Fsub[sorted_idx[-1], m] - Fsub[sorted_idx[0], m]
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (Fsub[sorted_idx[i + 1], m] - Fsub[sorted_idx[i - 1], m]) / f_range

            return dist

        def select_parents(F, indices):
            """Binary tournament selection."""
            i = rng.integers(len(indices))
            j = rng.integers(len(indices))
            # Use non-dominated rank and crowding distance
            return indices[i] if i < j else indices[j]

        def crossover(x1, x2):
            """Simulated binary crossover."""
            if rng.random() < 0.9:
                eta = 20
                beta1 = rng.random()
                beta2 = rng.random()
                if beta1 <= 0.5:
                    alpha = 2 * beta1
                    beta = (alpha ** (1 / (eta + 1)))
                else:
                    alpha = 2 * (1 - beta1)
                    beta = (alpha ** (1 / (eta + 1)))

                if beta2 <= 0.5:
                    alpha = 2 * beta2
                    beta = (alpha ** (1 / (eta + 1)))
                else:
                    alpha = 2 * (1 - beta2)
                    beta = (alpha ** (1 / (eta + 1)))

                child1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                child2 = 0.5 * ((1 + 1 / beta) * x1 + (1 - 1 / beta) * x2)
                return np.clip(child1, lo, hi), np.clip(child2, lo, hi)
            else:
                return x1.copy(), x2.copy()

        def mutate(x, rate=0.1):
            """Polynomial mutation."""
            if rng.random() < rate:
                eta = 20
                delta1 = rng.random()
                delta2 = rng.random()
                if delta1 <= 0.5:
                    delta = (2 * delta1) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - delta1)) ** (1 / (eta + 1))
                x_new = x + delta * (hi - lo)
                return np.clip(x_new, lo, hi)
            return x.copy()

        # NSGA-II main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = []
            for _ in range(pop_size // 2):
                if time.time() > deadline:
                    break
                # Tournament selection
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                p1 = X[i1] if i1 < i2 else X[i2]

                i3 = rng.integers(pop_size)
                i4 = rng.integers(pop_size)
                p2 = X[i3] if i3 < i4 else X[i4]

                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring_X.append(c1)
                offspring_X.append(c2)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)

            # Combine parents and offspring
            all_X = np.vstack([X, offspring_X])
            all_F = np.vstack([F, offspring_F])
            n_all = len(all_X)

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(all_F)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(all_X[idx])
                        next_F.append(all_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(all_F, front)
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in sorted_front[:remaining]:
                        idx = front[i]
                        next_X.append(all_X[idx])
                        next_F.append(all_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
