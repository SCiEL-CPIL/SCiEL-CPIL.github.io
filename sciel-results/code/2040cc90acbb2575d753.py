"""Multi-objective optimizer on WFG7 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Initial population
        pop_size = min(50, budget // 2)
        X = lo + range_ * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for dominance checking (use a large value)
        ref = np.full(n_obj, 1e10)

        # Helper: fast non-dominated sort using numpy
        def non_dominated_sort(F_arr):
            """Return list of fronts (list of indices)."""
            n = len(F_arr)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1

            # First front: points with domination_count == 0
            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front

            # Any remaining points (shouldn't happen for proper dominance)
            remaining = [i for i in range(n) if i not in [j for front in fronts for j in front]]
            if remaining:
                fronts.append(remaining)

            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            for m in range(n_obj):
                values = F_arr[indices, m]
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val - min_val < 1e-15:
                    continue
                idx_sorted = np.argsort(values)
                cd[idx_sorted[0]] = np.inf
                cd[idx_sorted[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[idx_sorted[i]] += (values[idx_sorted[i + 1]] - values[idx_sorted[i - 1]]) / (max_val - min_val)
            return cd

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Select parents from first front using tournament selection
            # Use binary tournament on first front
            front0 = fronts[0]
            if len(front0) >= 2:
                # Crowding distance for selection diversity
                cd = crowding_distance(F, front0)
                # Tournament: pick 2 random from front0, choose one with higher cd
                p1 = front0[rng.integers(len(front0))]
                p2 = front0[rng.integers(len(front0))]
                parent1_idx = p1 if cd[front0.index(p1)] >= cd[front0.index(p2)] else p2
            else:
                parent1_idx = front0[0]

            # Second parent: random from population
            parent2_idx = rng.integers(len(X))

            # SBX crossover
            child = X[parent1_idx].copy()
            if rng.random() < 0.9:  # crossover probability
                eta = 15.0
                for d in range(dim):
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                    child[d] = beta * X[parent1_idx, d] + (1 - beta) * X[parent2_idx, d]

            # Polynomial mutation
            if rng.random() < 0.9:  # mutation probability
                eta = 20.0
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u < 0.5:
                            delta_ = (2 * u) ** (1.0 / (eta + 1)) - 1
                        else:
                            delta_ = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                        child[d] += delta_ * range_[d]

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst point in last front
            last_front = fronts[-1]
            # Find worst by crowding distance within last front
            cd_last = crowding_distance(F, last_front)
            worst_local = np.argmin(cd_last)
            worst_global = last_front[worst_local]
            X[worst_global] = child
            F[worst_global] = fc

        return X
# EVOLVE-BLOCK-END
