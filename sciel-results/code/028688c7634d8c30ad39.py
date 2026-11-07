"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective DE/rand/1 with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a large constant, capped by budget
        pop_size = min(40, budget)
        if pop_size < 4:
            pop_size = 4

        rng = np.random.default_rng(0)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # DE parameters
        F_mut = 0.9
        CR = 0.9

        def fast_non_dominated_sort(F_arr):
            """Return list of fronts, each a list of indices."""
            n = len(F_arr)
            S = [[] for _ in range(n)]  # domination count
            n_dom = [0] * n  # number of solutions dominated
            fronts = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            current_front = [i for i in range(n) if n_dom[i] == 0]
            if not current_front:
                # All points dominate each other? Shouldn't happen, but fallback
                current_front = list(range(n))

            idx = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                idx += 1

            # If some points were never assigned (shouldn't happen), add them
            assigned = set()
            for front in fronts:
                for i in front:
                    assigned.add(i)
            if len(assigned) < n:
                remaining = [i for i in range(n) if i not in assigned]
                if remaining:
                    fronts.append(remaining)

            return fronts

        def crowding_distance(F_arr, front_indices):
            """Compute crowding distance for points in front_indices."""
            m = len(front_indices)
            if m <= 2:
                return [float('inf')] * m

            cd = [0.0] * m
            for k in range(n_obj):
                # Sort by objective k
                sorted_idx = sorted(range(m), key=lambda i: F_arr[front_indices[i]][k])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                f_min = F_arr[front_indices[sorted_idx[0]]][k]
                f_max = F_arr[front_indices[sorted_idx[-1]]][k]
                if f_max - f_min > 1e-15:
                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (
                            F_arr[front_indices[sorted_idx[i + 1]]][k] -
                            F_arr[front_indices[sorted_idx[i - 1]]][k]
                        ) / (f_max - f_min)
                else:
                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += float('inf')

            return cd

        # Main evolution loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Build mating pool: assign ranks and crowding distances
            rank = [0] * len(F)
            cd = [0.0] * len(F)

            for front_idx, front in enumerate(fronts):
                cd_front = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    rank[idx] = front_idx
                    cd[idx] = cd_front[i]

            # Generate offspring using DE/rand/1
            n_offspring = pop_size
            offspring_X = []
            offspring_F = []

            for i in range(n_offspring):
                # Check deadline
                if time.time() > deadline:
                    break

                # Select 3 distinct parents from population
                p1, p2, p3 = rng.choice(pop_size, size=3, replace=False)

                # DE/rand/1 mutation
                v = X[p1] + F_mut * (X[p2] - X[p3])
                v = np.clip(v, lo, hi)

                # Binary crossover
                trial = X[i].copy()
                jrand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                trial = np.clip(trial, lo, hi)

                # Evaluate
                ft = problem(trial)
                used += 1
                offspring_X.append(trial)
                offspring_F.append(ft)

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F)

            # Combine population and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            combined_size = len(combined_F)

            # Non-dominated sorting on combined
            comb_fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in comb_fronts:
                if len(new_X) + len(front) <= pop_size:
                    # Add entire front
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd_front = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    sorted_in_front = sorted(range(len(front)), key=lambda i: -cd_front[i])
                    for i in range(min(remaining, len(sorted_in_front))):
                        idx = front[sorted_in_front[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)
                pop_size = len(X)
            else:
                break

        return X
# EVOLVE-BLOCK-END
