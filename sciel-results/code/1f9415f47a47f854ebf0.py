"""Evolutionary multi-objective optimizer for WFG7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style evolutionary MOO with adaptive population sizing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Adaptive population size: smaller pop -> more generations
        # Cell findings: pop 20 gives better convergence than pop 30
        pop_size = 20

        # Initialize population
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size

        def non_dominated_sort(F):
            """Return (ranks, crowding_distances) for a set of objective vectors."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int), np.array([])

            ranks = np.zeros(n, dtype=int)
            crowding = np.zeros(n)

            front = set(range(n))
            rank = 0
            while len(front) > 0:
                front_list = list(front)
                nd_in_front = set()
                for i in front_list:
                    is_nd = True
                    for j in front_list:
                        if i == j:
                            continue
                        if np.all(F[i] >= F[j]) and np.any(F[i] > F[j]):
                            is_nd = False
                            break
                    if is_nd:
                        nd_in_front.add(i)

                for i in nd_in_front:
                    ranks[i] = rank

                if len(nd_in_front) > 1:
                    for k in range(n_obj):
                        sorted_front = sorted(nd_in_front, key=lambda x: F[x, k])
                        f_min = F[sorted_front[0], k]
                        f_max = F[sorted_front[-1], k]
                        if f_max <= f_min:
                            continue
                        for idx, i in enumerate(sorted_front):
                            if idx == 0 or idx == len(sorted_front) - 1:
                                crowding[i] = np.inf
                            else:
                                crowding[i] += (F[sorted_front[idx + 1], k] - F[sorted_front[idx - 1], k]) / (f_max - f_min)

                rank += 1
                front = front - nd_in_front

            return ranks, crowding

        def tournament_select(X, F, ranks, crowding, k=2):
            """Tournament selection based on non-dominated rank and crowding."""
            indices = np.random.randint(0, len(F), size=k)
            best = indices[0]
            for i in indices[1:]:
                if ranks[i] < ranks[best]:
                    best = i
                elif ranks[i] == ranks[best] and crowding[i] > crowding[best]:
                    best = i
            return X[best]

        def sbx_crossover(x1, x2, eta=20.0):
            """Simulated binary crossover."""
            x = np.zeros_like(x1)
            for i in range(dim):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        x[i] = x1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x1[i] - x2[i]))
                else:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        x[i] = x1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x1[i] - x2[i]))
            return np.clip(x, lo, hi)

        def polynomial_mutation(x, pmut=1.0 / dim, eta=20.0):
            """Polynomial mutation."""
            x_new = x.copy()
            for i in range(dim):
                if np.random.rand() < pmut:
                    range_i = hi[i] - lo[i]
                    if range_i < 1e-14:
                        continue
                    y = (x_new[i] - lo[i]) / range_i
                    if y < 0:
                        y = 0
                    if y > 1:
                        y = 1
                    u = np.random.rand()
                    if u < 0.5:
                        delta1 = (1 + 2 * y) ** (eta + 1)
                        delta = (1 / delta1) ** (1 / (eta + 1)) - 1
                        y_new = y + delta * (1 - y)
                    else:
                        delta2 = (1 + 2 * (1 - y)) ** (eta + 1)
                        delta = 1 - (1 / delta2) ** (1 / (eta + 1))
                        y_new = y + delta * y
                    x_new[i] = lo[i] + y_new * range_i
            return np.clip(x_new, lo, hi)

        # Main evolutionary loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            n_off = min(pop_size, self.budget - evaluations)
            X_off = []
            for _ in range(n_off):
                p1 = tournament_select(X, F, ranks, crowding)
                p2 = tournament_select(X, F, ranks, crowding)
                child = sbx_crossover(p1, p2)
                child = polynomial_mutation(child)
                X_off.append(child)

            X_off = np.array(X_off)
            F_off = np.array([problem(x) for x in X_off])
            evaluations += n_off

            # Combine parent and offspring
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])

            # Non-dominated sort and selection
            ranks, crowding = non_dominated_sort(F_all)
            sort_idx = np.lexsort(((-crowding), ranks))
            X = X_all[sort_idx[:pop_size]]
            F = F_all[sort_idx[:pop_size]]

        return np.zeros((1, dim))
# EVOLVE-BLOCK-END
