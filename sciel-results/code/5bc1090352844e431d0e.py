import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = min(50, max(20, budget // 4))
        n_gen = max(1, (budget - pop_size) // pop_size)

        # Initial population: mix of random and structured
        X = np.empty((pop_size, dim))
        # First half: uniform random
        X[:pop_size // 2] = lo + span * np.random.rand(pop_size // 2, dim)
        # Second half: random but with some structure (e.g., varying x1)
        for i in range(pop_size // 2, pop_size):
            X[i] = lo + span * np.random.rand(dim)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: non-dominated sorting
        def non_dominated_sort(F_arr):
            n = len(F_arr)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        n_count[i] += 1

            front = [i for i in range(n) if n_count[i] == 0]
            if not front:
                return [list(range(n))]

            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                front = next_front

            # Assign ranks
            ranks = np.zeros(n, dtype=int)
            for rank, f in enumerate(fronts):
                for idx in f:
                    ranks[idx] = rank
            return ranks

        # Helper: crowding distance
        def crowding_distance(F_arr, ranks):
            n = len(F_arr)
            cd = np.zeros(n)
            for rank in np.unique(ranks):
                idxs = np.where(ranks == rank)[0]
                if len(idxs) <= 2:
                    cd[idxs] = np.inf
                    continue
                for m in range(n_obj):
                    vals = F_arr[idxs, m]
                    min_v, max_v = np.min(vals), np.max(vals)
                    if max_v - min_v < 1e-15:
                        continue
                    sorted_idxs = idxs[np.argsort(vals)]
                    cd[sorted_idxs[0]] = np.inf
                    cd[sorted_idxs[-1]] = np.inf
                    for k in range(1, len(sorted_idxs) - 1):
                        cd[sorted_idxs[k]] += (vals[sorted_idxs[k + 1]] - vals[sorted_idxs[k - 1]]) / (max_v - min_v)
            return cd

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Evaluate offspring
            offspring_X = np.empty((pop_size, dim))
            offspring_F = np.empty((pop_size, n_obj))

            for i in range(pop_size):
                if time.time() > deadline:
                    # Fill remaining with copies of best
                    offspring_X[i] = X[0]
                    offspring_F[i] = F[0]
                    used += 1
                    continue

                # Select two parents via tournament selection
                # Combine ranks and crowding
                ranks = non_dominated_sort(F)
                cd = crowding_distance(F, ranks)

                def tournament():
                    a, b = np.random.randint(0, pop_size, 2)
                    if ranks[a] < ranks[b]:
                        return a
                    if ranks[b] < ranks[a]:
                        return b
                    return a if cd[a] > cd[b] else b

                p1 = tournament()
                p2 = tournament()

                # Crossover: simulated binary crossover
                child = np.empty(dim)
                for d in range(dim):
                    if np.random.rand() < 0.5:
                        # SBX
                        u = np.random.rand()
                        eta = 20
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        x1, x2 = X[p1, d], X[p2, d]
                        child[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    else:
                        child[d] = X[p1, d] if np.random.rand() < 0.5 else X[p2, d]

                # Mutation: polynomial mutation
                for d in range(dim):
                    if np.random.rand() < 0.1:
                        u = np.random.rand()
                        eta_m = 20
                        if u < 0.5:
                            delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                        child[d] += delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_n = 2 * pop_size

            # Non-dominated sort on combined
            combined_ranks = non_dominated_sort(combined_F)
            combined_cd = crowding_distance(combined_F, combined_ranks)

            # Select next generation
            next_X = np.empty((pop_size, dim))
            next_F = np.empty((pop_size, n_obj))
            count = 0
            for rank in range(combined_ranks.max() + 1):
                idxs = np.where(combined_ranks == rank)[0]
                if count + len(idxs) <= pop_size:
                    next_X[count:count + len(idxs)] = combined_X[idxs]
                    next_F[count:count + len(idxs)] = combined_F[idxs]
                    count += len(idxs)
                else:
                    # Sort by crowding distance descending
                    order = np.argsort(-combined_cd[idxs])
                    remaining = pop_size - count
                    selected = idxs[order[:remaining]]
                    next_X[count:pop_size] = combined_X[selected]
                    next_F[count:pop_size] = combined_F[selected]
                    count = pop_size
                    break

            X = next_X
            F = next_F

        return X
