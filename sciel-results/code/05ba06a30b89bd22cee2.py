"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility within the call
        rng = np.random.default_rng(42)

        # Initial population
        pop_size = min(40, budget // 2)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Reference point for dominance (we want to minimize both objectives)
        # We'll use a dynamic reference: the worst point in the current population
        # But for NSGA-II we just need dominance relations

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of points dominating i
            fronts = []
            front0 = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            for i in range(n):
                if n_dom[i] == 0:
                    front0.append(i)

            if front0:
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
                    fronts.append(next_front)
                    k += 1
                else:
                    break

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n, dtype=np.float64)
            F_sub = F[indices]

            for m in range(F_sub.shape[1]):
                order = np.argsort(F_sub[:, m])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                f_range = F_sub[order[-1], m] - F_sub[order[0], m]
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    dist[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / f_range

            return dist

        # Main loop
        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Combine with offspring
            # Create offspring of same size as population
            offspring_size = pop_size
            offspring_X = np.zeros((offspring_size, dim))
            offspring_F = np.zeros((offspring_size, self.n_obj))

            for i in range(offspring_size):
                if time.time() > deadline - 2.0:
                    break

                # Binary tournament selection
                idx1 = rng.integers(pop_size)
                idx2 = rng.integers(pop_size)

                # Get fronts for both
                fronts = fast_non_dominated_sort(F)
                # Find front rank
                front_rank = np.full(pop_size, -1, dtype=int)
                for rank, front in enumerate(fronts):
                    for idx in front:
                        front_rank[idx] = rank

                # Select parent with better front rank, or if same, higher crowding
                if front_rank[idx1] < front_rank[idx2]:
                    parent_idx = idx1
                elif front_rank[idx2] < front_rank[idx1]:
                    parent_idx = idx2
                else:
                    # Same front, use crowding distance
                    cd1 = crowding_distance(F, [idx1])[0]
                    cd2 = crowding_distance(F, [idx2])[0]
                    parent_idx = idx1 if cd1 > cd2 else idx2

                # Simulated binary crossover (SBX)
                if rng.random() < 0.9:
                    idx1 = parent_idx
                    idx2 = rng.integers(pop_size)
                    # Same front selection for second parent
                    if front_rank[idx1] < front_rank[idx2]:
                        pass
                    elif front_rank[idx2] < front_rank[idx1]:
                        pass
                    # Simple blend crossover
                    alpha = rng.random()
                    child = alpha * X[idx1] + (1 - alpha) * X[idx2]
                else:
                    child = X[parent_idx].copy()

                # Polynomial mutation
                if rng.random() < 0.8:
                    eta = 20.0
                    delta1 = (2.0 * rng.random() + 1.0) ** (1.0 / (eta + 1.0))
                    delta2 = (2.0 * rng.random() + 1.0) ** (1.0 / (eta + 1.0))
                    if rng.random() < 0.5:
                        delta = 1.0 - delta1
                    else:
                        delta = 1.0 - delta2
                    span = hi - lo
                    mutation = 0.5 * span * delta
                    child = child + mutation

                child = np.clip(child, lo, hi)
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            # Non-dominated sort on combined
            fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to trim this front using crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(-cd)  # descending crowding distance
                    for j in order[:remaining]:
                        idx = front[j]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
