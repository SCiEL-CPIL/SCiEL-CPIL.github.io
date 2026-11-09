"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer tuned for RE21 (budget=200)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Population size: 50 allows 4 generations of offspring (50*4=200)
        # Init (50) + Gen 1 (50) + Gen 2 (50) + Gen 3 (50) = 200
        # We will use a loop that stops when budget is reached.
        pop_size = 50
        if pop_size > self.budget:
            pop_size = self.budget

        # Initialize population
        X = lo + span * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Helper: Fast non-dominated sorting for 2 objectives
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            domination_count = np.zeros(n, dtype=int)
            dominated_by = [list() for _ in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if F[i, 0] <= F[j, 0] and F[i, 1] <= F[j, 1] and (F[i, 0] < F[j, 0] or F[i, 1] < F[j, 1]):
                        domination_count[j] += 1
                        dominated_by[i].append(j)
                    elif F[j, 0] <= F[i, 0] and F[j, 1] <= F[i, 1] and (F[j, 0] < F[i, 0] or F[j, 1] < F[i, 1]):
                        domination_count[i] += 1
                        dominated_by[j].append(i)

            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = set()
                for i in current_front:
                    for j in dominated_by[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.add(j)
                current_front = list(next_front)
            return fronts

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            F_sub = F[indices]
            for m in range(self.n_obj):
                order = np.argsort(F_sub[:, m])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                f_range = F_sub[order[-1], m] - F_sub[order[0], m]
                if f_range > 1e-15:
                    for k in range(1, n - 1):
                        dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / f_range
            return dist

        # SBX Crossover
        def sbx_crossover(p1, p2, eta=20.0):
            c1 = p1.copy()
            c2 = p2.copy()
            for d in range(self.dim):
                if np.random.random() < 0.5:
                    x1, x2 = p1[d], p2[d]
                    if abs(x1 - x2) > 1e-14:
                        if np.random.random() <= 1.0 / (eta + 1.0):
                            beta = 2.0 * np.random.random()
                            alpha = (beta + 1.0) ** (eta + 1.0)
                            beta_q = (1.0 / alpha) ** (1.0 / (eta + 1.0))
                        else:
                            beta = 1.0 / (2.0 - 2.0 * np.random.random())
                            alpha = (beta + 1.0) ** (eta + 1.0)
                            beta_q = (1.0 / alpha) ** (1.0 / (eta + 1.0))
                        c1[d] = 0.5 * ((1.0 + beta_q) * x1 + (1.0 - beta_q) * x2)
                        c2[d] = 0.5 * ((1.0 - beta_q) * x1 + (1.0 + beta_q) * x2)
            return c1, c2

        # Polynomial Mutation
        def poly_mutate(x, eta=20.0):
            x = x.copy()
            for d in range(self.dim):
                if np.random.random() < 1.0 / self.dim:
                    lo_d, hi_d = lo[d], hi[d]
                    if hi_d - lo_d < 1e-14:
                        continue
                    if np.random.random() <= 2.0 / (eta + 1.0):
                        beta = 1.0 + 2.0 * (x[d] - lo_d) / (hi_d - lo_d)
                        alpha = (beta + 1.0) ** (eta + 1.0)
                        delta_q = (1.0 / alpha) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        beta = 1.0 + 2.0 * (hi_d - x[d]) / (hi_d - lo_d)
                        alpha = (beta + 1.0) ** (eta + 1.0)
                        delta_q = 1.0 - (1.0 / alpha) ** (1.0 / (eta + 1.0))
                    x[d] = x[d] + (hi_d - lo_d) * delta_q
                    x[d] = np.clip(x[d], lo_d, hi_d)
            return x

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Calculate ranks
            fronts = non_dominated_sort(F)
            rank = np.zeros(pop_size, dtype=int)
            for front_idx, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = front_idx

            # Generate Offspring
            offspring_X = []
            offspring_F = []
            num_offspring = min(pop_size, self.budget - used)

            for _ in range(num_offspring // 2):  # Generate pairs
                if used + 2 > self.budget:
                    break
                # Tournament Selection
                i1, i2 = np.random.randint(pop_size, 2)
                p1_idx = i1 if rank[i1] <= rank[i2] else i2

                i3, i4 = np.random.randint(pop_size, 2)
                p2_idx = i3 if rank[i3] <= rank[i4] else i4

                # Crossover
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                c1 = poly_mutate(c1)
                c2 = poly_mutate(c2)

                offspring_X.append(c1)
                offspring_X.append(c2)

            # Evaluate Offspring
            for ox in offspring_X:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                of = problem(ox)
                offspring_F.append(of)
                used += 1

            if not offspring_F:
                break

            offspring_F = np.array(offspring_F, dtype=np.float64)
            offspring_X = np.array(offspring_X[:len(offspring_F)], dtype=np.float64)

            # Combine
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_size = len(combined_F)

            # Selection
            fronts = non_dominated_sort(combined_F)
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Crowding distance to break tie
                    remaining = pop_size - len(new_X)
                    dist = crowding_distance(combined_F, front)
                    # Sort by distance descending
                    sorted_indices = np.argsort(dist)[::-1]
                    for i in sorted_indices[:remaining]:
                        new_X.append(combined_X[front[i]])
                        new_F.append(combined_F[front[i]])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
