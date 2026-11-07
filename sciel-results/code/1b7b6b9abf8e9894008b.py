"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the evaluator's seeded RNG for reproducibility
        rng = np.random

        # Population size: balance exploration and exploitation
        pop_size = 40
        if budget < 100:
            pop_size = max(10, budget // 5)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume (used in scoring, but we just need good spread)
        ref_point = np.array([3.0, 5.0, 7.0])

        def non_dominated_sort(F):
            """NSGA-II non-dominated sorting. Returns list of fronts."""
            n = len(F)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1

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

            # Assign front indices
            front_index = np.full(n, -1, dtype=int)
            for idx, front in enumerate(fronts):
                for i in front:
                    front_index[i] = idx
            return fronts, front_index

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a given front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            distances = np.zeros(n)
            for m in range(n_obj):
                obj_vals = F[front_indices, m]
                min_val = obj_vals.min()
                max_val = obj_vals.max()
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = np.argsort(obj_vals)
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    distances[sorted_idx[k]] += (obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]) / (max_val - min_val)
            return distances

        # Main evolution loop
        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts, front_index = non_dominated_sort(F)

            # Create offspring
            offspring_X = []
            offspring_F = []

            n_offspring = pop_size
            for _ in range(n_offspring):
                if time.time() > deadline:
                    break

                # Selection via tournament on fronts
                if len(fronts) > 1:
                    # Prefer lower fronts
                    weights = [1.0 / (1.0 + fi) for fi in front_index]
                    weights = np.array(weights)
                    weights /= weights.sum()
                    parent_idx = rng.choice(pop_size, p=weights)
                else:
                    parent_idx = rng.randint(pop_size)

                parent = X[parent_idx]

                # Mutation: adaptive step size
                # Use a mix of small and large mutations
                if rng.random() < 0.7:
                    # Small perturbation
                    step = 0.05 * (hi - lo) * rng.standard_normal(dim)
                else:
                    # Larger exploration
                    step = 0.2 * (hi - lo) * rng.standard_normal(dim)

                child = parent + step
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

                if used >= budget:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            # Non-dominated sort on combined
            combined_fronts, combined_front_index = non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to truncate: use crowding distance
                    remaining = pop_size - len(new_X)
                    front_F = combined_F[front]
                    cd = crowding_distance(front_F, np.arange(len(front)))
                    sorted_by_cd = np.argsort(-cd)
                    for idx in sorted_by_cd[:remaining]:
                        new_X.append(combined_X[front[idx]])
                        new_F.append(combined_F[front[idx]])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
