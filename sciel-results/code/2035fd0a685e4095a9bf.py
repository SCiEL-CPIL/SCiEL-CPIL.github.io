"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive multi-objective optimizer for RE21 with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Use np.random (seeded by evaluator) for reproducibility
        # Initial population
        pop_size = 20
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F_arr):
            """Return list of fronts (lists of indices)."""
            n = len(F_arr)
            S = [[] for _ in range(n)]  # set of solutions dominated by i
            ndom = [0] * n
            rank = [0] * n

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        ndom[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].append(i)
                        ndom[i] += 1

            fronts = []
            current_front = [i for i in range(n) if ndom[i] == 0]
            for i in current_front:
                rank[i] = 0
            if current_front:
                fronts.append(current_front)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        ndom[j] -= 1
                        if ndom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            m = F_arr.shape[1]
            for k in range(m):
                obj = F_arr[indices, k]
                min_val = obj.min()
                max_val = obj.max()
                if max_val - min_val < 1e-12:
                    cd[:] = np.inf
                    continue
                order = np.argsort(obj)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (obj[order[i + 1]] - obj[order[i - 1]]) / (max_val - min_val)
            return cd

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline - 5:
                break

            # Evaluate current population
            fronts = fast_non_dominated_sort(F)

            # Build mating pool from first fronts
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) > pop_size:
                    # Need crowding to select
                    cd = crowding_distance(F, front)
                    order = np.argsort(-cd)
                    mating_pool.extend(front[i] for i in order[:pop_size - len(mating_pool)])
                    break
                else:
                    mating_pool.extend(front)

            # Generate offspring via crossover and mutation
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, budget - used)

            for _ in range(n_offspring):
                if time.time() > deadline - 5:
                    break
                # Select two parents from mating pool
                p1_idx = mating_pool[np.random.randint(len(mating_pool))]
                p2_idx = mating_pool[np.random.randint(len(mating_pool))]
                p1 = X[p1_idx]
                p2 = X[p2_idx]

                # SBX crossover
                if np.random.rand() < 0.9:
                    eta = 20
                    child1 = np.zeros(dim)
                    child2 = np.zeros(dim)
                    for d in range(dim):
                        if np.random.rand() < 0.5:
                            u = np.random.rand()
                            if u <= 1 / (eta + 1):
                                beta = (2 * u) ** (1 / (eta + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                            child1[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                            child2[d] = 0.5 * ((1 - beta) * p1[d] + (1 + beta) * p2[d])
                        else:
                            child1[d] = p1[d]
                            child2[d] = p2[d]
                else:
                    child1 = p1.copy()
                    child2 = p2.copy()

                # Polynomial mutation
                for d in range(dim):
                    if np.random.rand() < 1.0 / dim:
                        u = np.random.rand()
                        if u < 0.5:
                            delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                        else:
                            delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                        child1[d] += delta_q * (hi[d] - lo[d])
                    if np.random.rand() < 1.0 / dim:
                        u = np.random.rand()
                        if u < 0.5:
                            delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                        else:
                            delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                        child2[d] += delta_q * (hi[d] - lo[d])

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                offspring_X.append(child1)
                offspring_F.append(problem(child1))
                used += 1

                if used >= budget or time.time() > deadline - 5:
                    break

                offspring_X.append(child2)
                offspring_F.append(problem(child2))
                used += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine and select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            fronts = fast_non_dominated_sort(combined_F)

            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.extend(combined_X[i] for i in front)
                    new_F.extend(combined_F[i] for i in front)
                else:
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(-cd)
                    need = pop_size - len(new_X)
                    new_X.extend(combined_X[front[i]] for i in order[:need])
                    new_F.extend(combined_F[front[i]] for i in order[:need])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
