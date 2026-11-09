"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = 40
        # Reserve some evaluations for initial population
        init_evals = pop_size
        if init_evals >= budget:
            pop_size = max(4, budget - 1)
            init_evals = pop_size

        # Initialize population
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = init_evals

        def non_dominated_sort(F):
            """Return list of fronts, each front is list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = []
            front0 = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1
                if n_dom[i] == 0:
                    front0.append(i)
            fronts.append(front0)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, indices):
            """Return crowding distances for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                rng = vals[order[-1]] - vals[order[0]]
                if rng > 1e-15:
                    for i in range(1, n - 1):
                        cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / rng
            return cd

        def sbx_crossover(x1, x2, lo, hi, eta=20.0):
            """Simulated Binary Crossover."""
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(dim):
                if np.random.rand() < 0.5:
                    x1i, x2i = x1[i], x2[i]
                    if abs(x1i - x2i) < 1e-15:
                        continue
                    if x1i > x2i:
                        x1i, x2i = x2i, x1i
                    lo_i, hi_i = lo[i], hi[i]
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    c1 = 0.5 * ((1 + beta) * x1i + (1 - beta) * x2i)
                    c2 = 0.5 * ((1 - beta) * x1i + (1 + beta) * x2i)
                    # Clip to bounds
                    if c1 < lo_i or c1 > hi_i:
                        c1 = x1[i]
                    if c2 < lo_i or c2 > hi_i:
                        c2 = x2[i]
                    child1[i] = c1
                    child2[i] = c2
            return child1, child2

        def poly_mutate(x, lo, hi, eta=20.0, prob=0.1):
            """Polynomial mutation."""
            x = x.copy()
            for i in range(dim):
                if np.random.rand() < prob:
                    x_i = x[i]
                    lo_i, hi_i = lo[i], hi[i]
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    x[i] = x_i + delta * (hi_i - lo_i)
                    x[i] = np.clip(x[i], lo_i, hi_i)
            return x

        def tournament_select(pop_F, pop_X, fronts, k=2):
            """Binary tournament on rank + crowding distance."""
            # Assign rank
            rank = np.zeros(len(pop_F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            # Compute crowding distance per front
            cd = np.zeros(len(pop_F))
            for front in fronts:
                if len(front) > 2:
                    cd[front] = crowding_distance(pop_F, front)
            # Tournament
            i1 = np.random.randint(len(pop_F))
            i2 = np.random.randint(len(pop_F))
            if rank[i1] < rank[i2]:
                return pop_X[i1].copy()
            elif rank[i2] < rank[i1]:
                return pop_X[i2].copy()
            else:
                if cd[i1] > cd[i2]:
                    return pop_X[i1].copy()
                else:
                    return pop_X[i2].copy()

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_off = pop_size
            for _ in range(n_off):
                if time.time() > deadline:
                    break
                p1 = tournament_select(F, X, fronts)
                p2 = tournament_select(F, X, fronts)
                c1, c2 = sbx_crossover(p1, p2, lo, hi)
                c1 = poly_mutate(c1, lo, hi)
                c2 = poly_mutate(c2, lo, hi)
                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(problem(c1))
                offspring_F.append(problem(c2))
                used += 2
                if used >= budget:
                    break
            if not offspring_X:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            n_combined = len(combined_F)

            # Non-dominated sort combined
            fronts_c = non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts_c:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to truncate this front by crowding distance
                    remaining = pop_size - len(new_X)
                    if remaining <= 0:
                        break
                    cd_front = crowding_distance(combined_F, front)
                    order = np.argsort(-cd_front)
                    for j in order[:remaining]:
                        idx = front[j]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)
                pop_size = len(X)

            # If we ran out of budget, break
            if used >= budget:
                break

        return X
# EVOLVE-BLOCK-END
