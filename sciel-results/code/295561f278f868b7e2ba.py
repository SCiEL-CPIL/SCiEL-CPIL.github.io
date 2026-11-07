"""Multi-objective optimizer for DTLZ1."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(80, budget // 2)
        if pop_size < 10:
            pop_size = 10

        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Track all evaluated points for hypervolume
        all_points = list(F)

        # NSGA-II helper functions
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_dominates = [0] * n
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1
                if n_dominates[i] == 0:
                    fronts[0].append(i)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            return fronts

        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return [float('inf')] * len(front_indices)
            n = len(front_indices)
            dist = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[front_indices, m]
                min_val = f_vals.min()
                max_val = f_vals.max()
                if max_val == min_val:
                    continue
                sorted_idx = np.argsort(f_vals)
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (f_vals[sorted_idx[i + 1]] - f_vals[sorted_idx[i - 1]]) / (max_val - min_val)
            return dist

        def tournament_select(pop, fronts, crowding, k=2):
            indices = []
            for _ in range(k):
                idx = rng.integers(len(pop))
                indices.append(idx)
            best = indices[0]
            for idx in indices[1:]:
                fi = next(f for f in fronts if idx in f)
                fb = next(f for f in fronts if best in f)
                if fi < fb:
                    best = idx
                elif fi == fb:
                    if crowding[idx] > crowding[best]:
                        best = idx
            return pop[best]

        # Main evolutionary loop
        while used < budget:
            # Check time
            if time.time() > deadline - 2:
                break

            # Get fronts and crowding
            fronts = fast_non_dominated_sort(F)
            
            # Compute crowding distance per front
            crowding = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowding[idx] = cd[i]

            # Create offspring
            offspring_count = min(pop_size, budget - used)
            if offspring_count <= 0:
                break

            # Adaptive mutation rate
            progress = used / budget
            mutation_scale = 0.3 * (1.0 - 0.5 * progress)
            crossover_prob = 0.8

            offspring_X = []
            offspring_F = []
            for _ in range(offspring_count):
                # Tournament selection for two parents
                p1 = tournament_select(X, fronts, crowding)
                p2 = tournament_select(X, fronts, crowding)

                if rng.random() < crossover_prob:
                    # Simulated binary crossover
                    child = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            child[d] = p1[d]
                        else:
                            child[d] = p2[d]
                    # Add some blend
                    alpha = rng.random()
                    child = alpha * p1 + (1 - alpha) * p2
                else:
                    child = p1.copy()

                # Mutation
                if rng.random() < 0.9:
                    mutation = mutation_scale * (hi - lo) * rng.standard_normal(dim)
                    child = child + mutation
                    child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
                all_points.append(fc)

            if used >= budget:
                break

            # Combine parents and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            combined_size = len(combined_F)

            # Non-dominated sort on combined
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to trim this front
                    remaining = pop_size - len(new_X)
                    # Compute crowding distance for this front
                    cd = crowding_distance(combined_F, front)
                    sorted_idx = np.argsort(-cd)
                    for i in range(min(remaining, len(front))):
                        idx = front[sorted_idx[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
