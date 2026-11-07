"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II with SBX and polynomial mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        pop_size = 40
        n_evals = 0

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        n_evals += pop_size

        # NSGA-II parameters
        crossover_prob = 0.9
        mutation_prob = 0.2
        eta_c = 20.0  # SBX distribution index
        eta_m = 20.0  # Polynomial mutation distribution index

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort, returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            current_front = []
            for i in range(n):
                if n_dom[i] == 0:
                    rank[i] = 0
                    current_front.append(i)

            fronts.append(current_front)
            k = 0
            while len(current_front) > 0:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                current_front = next_front

            return fronts, rank

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(self.n_obj):
                obj_vals = F[front_indices, m]
                min_val = obj_vals.min()
                max_val = obj_vals.max()
                if max_val - min_val < 1e-15:
                    dist[:] = np.inf
                    continue
                sorted_idx = np.argsort(obj_vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[sorted_idx[k]] += (obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]) / (max_val - min_val)
            return dist

        def sbx_crossover(x1, x2, lo, hi, rng, crossover_prob, eta_c):
            """Simulated Binary Crossover."""
            if rng.random() > crossover_prob:
                return x1.copy(), x2.copy()

            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(len(x1)):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        continue
                    if rng.random() < 0.5:
                        u = rng.random()
                        if u <= 1.0 / (eta_c + 1):
                            beta = (2 * u) ** (1.0 / (eta_c + 1))
                        else:
                            beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta_c + 1))
                        child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    else:
                        u = rng.random()
                        if u <= 1.0 / (eta_c + 1):
                            beta = (2 * u) ** (1.0 / (eta_c + 1))
                        else:
                            beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta_c + 1))
                        child1[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                        child2[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])

            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            return child1, child2

        def polynomial_mutation(x, lo, hi, rng, mutation_prob, eta_m):
            """Polynomial Mutation."""
            x_mut = x.copy()
            for i in range(len(x)):
                if rng.random() < mutation_prob:
                    if x_mut[i] <= lo[i]:
                        continue
                    if x_mut[i] >= hi[i]:
                        continue
                    delta1 = (x_mut[i] - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - x_mut[i]) / (hi[i] - lo[i])
                    u = rng.random()
                    if u < 0.5:
                        xy = (1 + 2 * (1 - delta1)) ** (-(eta_m + 1))
                        deltaq = (xy - 1) if (2 * u - 1) < 0 else 1 - (xy - 1)
                    else:
                        xy = (1 + 2 * delta2) ** (-(eta_m + 1))
                        deltaq = (xy - 1) if (2 * u - 1) < 0 else 1 - (xy - 1)
                    x_mut[i] += deltaq * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main loop
        while n_evals < self.budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = []
            for _ in range(pop_size):
                # Tournament selection
                i, j = rng.integers(0, pop_size, size=2)
                if i == j:
                    continue
                # Compare ranks, then crowding distance
                fronts, rank = fast_non_dominated_sort(F)
                if rank[i] < rank[j]:
                    parent1_idx = i
                elif rank[j] < rank[i]:
                    parent1_idx = j
                else:
                    # Same rank, use crowding distance
                    front_idx = [k for k, f in enumerate(fronts) if i in f]
                    if len(front_idx) == 0:
                        parent1_idx = i
                    else:
                        fi = front_idx[0]
                        cd = crowding_distance(F, fronts[fi])
                        cd_map = {idx: cd[k] for k, idx in enumerate(fronts[fi])}
                        if cd_map.get(i, 0) >= cd_map.get(j, 0):
                            parent1_idx = i
                        else:
                            parent1_idx = j

                # Second parent
                i2, j2 = rng.integers(0, pop_size, size=2)
                if i2 == j2:
                    continue
                if rank[i2] < rank[j2]:
                    parent2_idx = i2
                elif rank[j2] < rank[i2]:
                    parent2_idx = j2
                else:
                    front_idx2 = [k for k, f in enumerate(fronts) if i2 in f]
                    if len(front_idx2) == 0:
                        parent2_idx = i2
                    else:
                        fi2 = front_idx2[0]
                        cd2 = crowding_distance(F, fronts[fi2])
                        cd_map2 = {idx: cd2[k] for k, idx in enumerate(fronts[fi2])}
                        if cd_map2.get(i2, 0) >= cd_map2.get(j2, 0):
                            parent2_idx = i2
                        else:
                            parent2_idx = j2

                # Crossover
                child1, child2 = sbx_crossover(X[parent1_idx], X[parent2_idx], lo, hi, rng, crossover_prob, eta_c)

                # Mutation
                child1 = polynomial_mutation(child1, lo, hi, rng, mutation_prob, eta_m)
                child2 = polynomial_mutation(child2, lo, hi, rng, mutation_prob, eta_m)

                offspring_X.append(child1)
                offspring_X.append(child2)

                n_evals += 2
                if n_evals >= self.budget:
                    break

                if time.time() > deadline:
                    break

            if len(offspring_X) == 0:
                break

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort
            fronts, rank = fast_non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to truncate this front by crowding distance
                    cd = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(-cd)
                    needed = pop_size - len(new_X)
                    for k in range(needed):
                        idx = front[sorted_front[k]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
