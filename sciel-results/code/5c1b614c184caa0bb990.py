"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using NSGA-II-like approach with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Population size: keep small for efficiency with 200 budget
        pop_size = 20
        used = 0

        # Initial population: random
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used += pop_size

        # Store all evaluated points for hypervolume
        all_F = F.copy()

        def fast_non_dominated_sort(F_arr):
            """Return list of fronts (each a list of indices)."""
            n = len(F_arr)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1

            front = [i for i in range(n) if domination_count[i] == 0]
            fronts.append(front)

            while front:
                next_front = []
                for i in front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                front = next_front
                if front:
                    fronts.append(front)

            return fronts

        def crowding_distance(F_arr, front_indices):
            """Compute crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                f_vals = F_arr[front_indices, m]
                f_min = f_vals.min()
                f_max = f_vals.max()
                if f_max == f_min:
                    continue
                idx_sorted = np.argsort(f_vals)
                cd[idx_sorted[0]] = np.inf
                cd[idx_sorted[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[idx_sorted[k]] += (f_vals[idx_sorted[k + 1]] - f_vals[idx_sorted[k - 1]]) / (f_max - f_min)
            return cd

        def sbx_crossover(x1, x2):
            """Simulated binary crossover."""
            eta_c = 15.0
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(dim):
                if rng.random() < 0.5:
                    if x1[i] == x2[i]:
                        continue
                    if x1[i] > x2[i]:
                        x1[i], x2[i] = x2[i], x1[i]
                    if rng.random() < 0.5:
                        beta = (2 * rng.random()) ** (1 / (eta_c + 1))
                        child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    else:
                        beta = (2 * (1 - rng.random())) ** (1 / (eta_c + 1))
                        child2[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
            return child1, child2

        def polynomial_mutation(x):
            """Polynomial mutation."""
            eta_m = 20.0
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < 0.5:
                    lo_i, hi_i = lo[i], hi[i]
                    delta = rng.random()
                    if delta < 0.5:
                        val = (2 * delta) ** (1 / (eta_m + 1))
                        x_mut[i] = x[i] + (hi_i - lo_i) * (val - 1)
                    else:
                        val = (2 * (1 - delta)) ** (1 / (eta_m + 1))
                        x_mut[i] = x[i] + (hi_i - lo_i) * (1 - val)
                    x_mut[i] = np.clip(x_mut[i], lo_i, hi_i)
            return x_mut

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Get fronts
            fronts = fast_non_dominated_sort(F)
            front0 = fronts[0] if fronts else list(range(pop_size))

            # Select parents from first front using crowding distance
            cd = crowding_distance(F, front0)
            # Tournament selection: pick 2 random, choose one with higher crowding distance
            def select_parent():
                i1 = rng.integers(len(front0))
                i2 = rng.integers(len(front0))
                if cd[i1] >= cd[i2]:
                    return front0[i1]
                else:
                    return front0[i2]

            p1 = select_parent()
            p2 = select_parent()

            # Crossover and mutation
            child1, child2 = sbx_crossover(X[p1], X[p2])
            child1 = polynomial_mutation(child1)
            child2 = polynomial_mutation(child2)

            # Evaluate children
            fc1 = problem(child1)
            used += 1
            all_F = np.vstack([all_F, fc1])
            if used >= self.budget or time.time() > deadline:
                break

            fc2 = problem(child2)
            used += 1
            all_F = np.vstack([all_F, fc2])

            # Combine population
            X_new = np.vstack([X, child1, child2])
            F_new = np.vstack([F, fc1, fc2])

            # Truncate to pop_size using non-dominated sorting + crowding
            fronts_new = fast_non_dominated_sort(F_new)
            X_selected = []
            F_selected = []
            for front in fronts_new:
                if len(X_selected) + len(front) <= pop_size:
                    X_selected.append(X_new[front])
                    F_selected.append(F_new[front])
                else:
                    remaining = pop_size - len(X_selected)
                    cd_front = crowding_distance(F_new, front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd_front)
                    selected_idx = [front[i] for i in order[:remaining]]
                    X_selected.append(X_new[selected_idx])
                    F_selected.append(F_new[selected_idx])
                    break

            if X_selected:
                X = np.vstack(X_selected)
                F = np.vstack(F_selected)
            else:
                X = X_new[:pop_size]
                F = F_new[:pop_size]

        return X
# EVOLVE-BLOCK-END
