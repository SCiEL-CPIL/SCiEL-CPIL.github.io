"""Multi-objective optimizer on WFG4 using a refined evolutionary strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer with adaptive mutation and time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Population size
        pop_size = 50

        # Initial random population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best hypervolume points
        all_points = list(F)

        def fast_non_dominated_sort(F_arr):
            """Return non-dominated indices."""
            n_pts = len(F_arr)
            if n_pts == 0:
                return np.array([], dtype=int)
            nd_indices = []
            for i in range(n_pts):
                dominated = False
                for j in range(n_pts):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return np.array(nd_indices, dtype=int)

        def crowding_distance(F_arr, nd_indices):
            """Compute crowding distance for non-dominated points."""
            if len(nd_indices) <= 2:
                return np.ones(len(nd_indices))
            distances = np.zeros(len(nd_indices))
            for k in range(m):
                sorted_idx = np.argsort(F_arr[nd_indices, k])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                f_range = F_arr[nd_indices, k].max() - F_arr[nd_indices, k].min()
                if f_range > 1e-15:
                    for i in range(1, len(nd_indices) - 1):
                        distances[sorted_idx[i]] += (
                            F_arr[nd_indices[sorted_idx[i + 1]], k] -
                            F_arr[nd_indices[sorted_idx[i - 1]], k]
                        ) / f_range
            return distances

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_idx = fast_non_dominated_sort(F)
            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(np.sum(F, axis=1))])

            # Compute crowding distances
            cd = crowding_distance(F, nd_idx)

            # Select parents from non-dominated set with crowding
            # Tournament selection: pick 2 random from nd set, choose higher crowding
            def select_parent():
                i, j = rng.integers(len(nd_idx), size=2)
                if cd[i] > cd[j]:
                    return nd_idx[i]
                elif cd[j] > cd[i]:
                    return nd_idx[j]
                else:
                    return nd_idx[rng.integers(2)]

            # Adaptive mutation step
            # Use a smaller step as we converge
            progress = used / self.budget
            base_step = 0.15 * (1.0 - 0.5 * progress)

            # Generate children
            n_children = 10
            for _ in range(n_children):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                p1 = select_parent()
                p2 = select_parent()

                # SBX crossover
                if rng.random() < 0.9:
                    # Simulated binary crossover
                    child1 = np.zeros(n)
                    child2 = np.zeros(n)
                    for d in range(n):
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / 20)
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / 20)
                        child1[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                        child2[d] = 0.5 * ((1 - beta) * X[p1, d] + (1 + beta) * X[p2, d])
                    child = child1 if rng.random() < 0.5 else child2
                else:
                    child = X[p1].copy()

                # Mutation
                mutation_rate = 0.3
                for d in range(n):
                    if rng.random() < mutation_rate:
                        child[d] += base_step * (hi[d] - lo[d]) * rng.standard_normal()
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                all_points.append(fc)

                # Replace worst point in population
                # Use a simple heuristic: replace point with highest sum of objectives
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = child
                F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
