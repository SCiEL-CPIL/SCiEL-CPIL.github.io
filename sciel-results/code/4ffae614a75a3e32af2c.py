"""Multi-objective optimizer for WFG7 with deadline guard and efficient NSGA-II."""
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(50, max(10, budget // 4))

        # Random initialization
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size

        def is_deadline():
            return time.time() >= deadline

        def fast_non_dominated_sort(F):
            """Return list of fronts (indices) and rank array."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominated = np.zeros(n, dtype=int)
            rank = np.full(n, -1, dtype=int)
            front0 = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominated[i] += 1
                if n_dominated[i] == 0:
                    rank[i] = 0
                    front0.append(i)

            fronts = [front0]
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                    k += 1
            return fronts, rank

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[indices, m]
                f_min, f_max = f_vals.min(), f_vals.max()
                if f_max - f_min < 1e-15:
                    continue
                order = np.argsort(f_vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (f_vals[order[i + 1]] - f_vals[order[i - 1]]) / (f_max - f_min)
            return cd

        # Main loop
        while evals < budget:
            if is_deadline():
                break

            # Fast non-dominated sorting
            fronts, rank = fast_non_dominated_sort(F)

            # Build mating pool from first fronts
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) >= pop_size:
                    # Need crowding distance for this front
                    cd = crowding_distance(F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    mating_pool.extend(sorted_front[:pop_size - len(mating_pool)])
                    break
                else:
                    mating_pool.extend(front)

            if not mating_pool:
                mating_pool = list(range(len(F)))

            # Generate child via tournament selection + crossover + mutation
            # Binary tournament
            i1, i2 = np.random.randint(0, len(mating_pool), 2)
            p1_idx = mating_pool[i1] if rank[p1_idx] < rank[mating_pool[i2]] else mating_pool[i2]
            # Second parent
            i3, i4 = np.random.randint(0, len(mating_pool), 2)
            p2_idx = mating_pool[i3] if rank[p1_idx] < rank[mating_pool[i4]] else mating_pool[i4]

            p1 = X[p1_idx]
            p2 = X[p2_idx]

            # Simulated binary crossover (SBX)
            if np.random.random() < 0.9:
                eta = 20.0
                child = np.zeros(dim)
                for d in range(dim):
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                    child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
            else:
                child = p1.copy()

            # Polynomial mutation
            if np.random.random() < 0.9:
                eta_m = 20.0
                for d in range(dim):
                    if np.random.random() < 1.0 / dim:
                        u = np.random.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                        child[d] += delta * (hi[d] - lo[d])

            child = np.clip(child, lo, hi)

            # Evaluate
            fc = problem(child)
            evals += 1

            # Replace worst point
            # Find worst by rank, then crowding distance
            worst_idx = np.argmax(rank)
            # Among points with same rank, pick the one with lowest crowding distance
            same_rank = np.where(rank == rank[worst_idx])[0]
            if len(same_rank) > 1:
                cd_same = crowding_distance(F, same_rank)
                worst_local = np.argmin(cd_same)
                worst_idx = same_rank[worst_local]

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
