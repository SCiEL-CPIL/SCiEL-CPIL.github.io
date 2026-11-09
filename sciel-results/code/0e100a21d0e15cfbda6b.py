"""Multi-objective optimizer for WFG9 using NSGA-II with adaptive parameters."""
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size
        pop_size = max(20, min(60, budget // 3))

        # Initialize population with LHS
        n_init = min(pop_size, budget)
        perm = np.argsort(np.argsort(np.random.rand(n_init, dim)), axis=0)
        u = (perm + np.random.rand(n_init, dim)) / n_init
        population = lo + range_ * u

        # Evaluate initial population
        used = 0
        F = np.zeros((n_init, n_obj))
        for i in range(n_init):
            if time.time() > deadline:
                break
            f = problem(population[i])
            used += 1
            if np.any(np.isnan(f)) or np.any(np.isinf(f)):
                f = np.full(n_obj, 1e10)
            F[i] = f

        # Helper functions for NSGA-II
        def fast_non_dominated_sort(F):
            """
            Fast non-dominated sorting.
            Returns: list of fronts, where each front is a list of indices
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominates = np.zeros(n, dtype=int)  # number of points i dominates
            front = []

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominates[j] += 1

            for i in range(n):
                if n_dominates[i] == 0:
                    front.append(i)

            fronts = [front]
            i = 0
            while fronts[i]:
                next_front = []
                for k in fronts[i]:
                    for m in S[k]:
                        n_dominates[m] -= 1
                        if n_dominates[m] == 0:
                            next_front.append(m)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance_assignment(F, indices):
            """
            Assign crowding distances to a set of indices in F.
            Returns: dict mapping index -> crowding distance
            """
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}

            n = len(indices)
            dist = np.zeros(n)
            F_sub = F[indices]

            for m in range(n_obj):
                # Sort by m-th objective
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 0:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (
                            F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]
                        ) / f_range

            result = {}
            for i, idx in enumerate(indices):
                result[idx] = dist[i]
            return result

        def tournament_selection(pop_indices, ranks, crowding, k=2):
            """
            Tournament selection.
            pop_indices: list of all population indices
            ranks: dict index -> rank
            crowding: dict index -> crowding distance
            Returns: selected index
            """
            candidates = np.random.choice(len(pop_indices), size=k, replace=False)
            best = candidates[0]
            for c in candidates[1:]:
                idx_best = pop_indices[best]
                idx_c = pop_indices[c]
                rank_best = ranks.get(idx_best, 999999)
                rank_c = ranks.get(idx_c, 999999)
                if rank_c < rank_best:
                    best = c
                elif rank_c == rank_best:
                    cd_best = crowding.get(idx_best, 0)
                    cd_c = crowding.get(idx_c, 0)
                    if cd_c > cd_best:
                        best = c
            return pop_indices[best]

        def SBX_crossover(x1, x2, lo, hi, dim, eta=20.0):
            """
            Simulated Binary Crossover.
            Returns two children.
            """
            c1 = np.zeros(dim)
            c2 = np.zeros(dim)
            for i in range(dim):
                if np.random.rand() < 0.5:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
                else:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        if np.random.rand() < 0.5:
                            beta = 1.0 / (1.0 + 2.0 * np.random.rand())
                            beta_q = (0.5 * (1.0 + (1.0 + 2.0 * (x1[i] - lo[i]) /
                                                    (x2[i] - lo[i] + 1e-14))) ** (eta + 1.0))
                            c1[i] = 0.5 * ((1.0 + beta_q) * x1[i] + (1.0 - beta_q) * x2[i])
                            c2[i] = 0.5 * ((1.0 + 1.0 / beta_q) * x2[i] + (1.0 - 1.0 / beta_q) * x1[i])
                        else:
                            beta = 1.0 / (1.0 + 2.0 * np.random.rand())
                            beta_q = (0.5 * (1.0 + (1.0 + 2.0 * (x2[i] - lo[i]) /
                                                    (x1[i] - lo[i] + 1e-14))) ** (eta + 1.0))
                            c1[i] = 0.5 * ((1.0 + beta_q) * x2[i] + (1.0 - beta_q) * x1[i])
                            c2[i] = 0.5 * ((1.0 + 1.0 / beta_q) * x1[i] + (1.0 - 1.0 / beta_q) * x2[i])
                    else:
                        c1[i] = x1[i]
                        c2[i] = x2[i]
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            return c1, c2

        def polynomial_mutation(x, lo, hi, dim, eta=20.0, prob_mut=None):
            """
            Polynomial mutation.
            """
            if prob_mut is None:
                prob_mut = 1.0 / dim
            x = x.copy()
            for i in range(dim):
                if np.random.rand() < prob_mut:
                    y = x[i]
                    if y <= lo[i] + 1e-14:
                        y_new = hi[i]
                    elif y >= hi[i] - 1e-14:
                        y_new = lo[i]
                    else:
                        r = np.random.rand()
                        if r <= 0.5:
                            xy = (y - lo[i]) / (hi[i] - lo[i] + 1e-14)
                            mut_pow = 2.0 / (eta + 1.0)
                            val = 1.0 - (1.0 - xy) ** (eta + 1.0)
                            if r * val <= 0.5:
                                delta_q = (2.0 * r) ** (1.0 / (eta + 1.0))
                            else:
                                delta_q = (2.0 * (1.0 - r)) ** (1.0 / (eta + 1.0))
                            y_new = y + (hi[i] - lo[i]) * (delta_q - 1.0)
                        else:
                            xy = (hi[i] - y) / (hi[i] - lo[i] + 1e-14)
                            mut_pow = 2.0 / (eta + 1.0)
                            val = 1.0 - (1.0 - xy) ** (eta + 1.0)
                            if (1.0 - r) * val <= 0.5:
                                delta_q = (2.0 * (1.0 - r)) ** (1.0 / (eta + 1.0))
                            else:
                                delta_q = (2.0 * r) ** (1.0 / (eta + 1.0))
                            y_new = y - (hi[i] - lo[i]) * (delta_q - 1.0)
                    x[i] = np.clip(y_new, lo[i], hi[i])
            return x

        # Main NSGA-II loop
        current_pop = population.copy()
        current_F = F.copy()
        current_indices = list(range(n_init))

        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring population
            offspring = []
            offspring_F = []

            # Get ranks and crowding distances for current population
            fronts = fast_non_dominated_sort(current_F)
            ranks = {}
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            crowding = {}
            for front in fronts:
                cd = crowding_distance_assignment(current_F, front)
                crowding.update(cd)

            n_offspring = len(current_indices)
            for _ in range(n_offspring // 2):
                if time.time() > deadline or used + 2 > budget:
                    break

                # Select two parents
                parent1_idx = tournament_selection(current_indices, ranks, crowding)
                parent2_idx = tournament_selection(current_indices, ranks, crowding)

                # Crossover
                c1, c2 = SBX_crossover(current_pop[parent1_idx], current_pop[parent2_idx],
                                       lo, hi, dim, eta=20.0)

                # Mutation
                c1 = polynomial_mutation(c1, lo, hi, dim, eta=20.0, prob_mut=1.0/dim)
                c2 = polynomial_mutation(c2, lo, hi, dim, eta=20.0, prob_mut=1.0/dim)

                # Evaluate
                f1 = problem(c1)
                used += 1
                if np.any(np.isnan(f1)) or np.any(np.isinf(f1)):
                    f1 = np.full(n_obj, 1e10)
                f2 = problem(c2)
                used += 1
                if np.any(np.isnan(f2)) or np.any(np.isinf(f2)):
                    f2 = np.full(n_obj, 1e10)

                offspring.append(c1)
                offspring.append(c2)
                offspring_F.append(f1)
                offspring_F.append(f2)

            if not offspring:
                break

            # Combine parent and offspring
            combined = np.vstack([current_pop, np.array(offspring)])
            combined_F = np.vstack([current_F, np.array(offspring_F)])
            combined_indices = list(range(len(combined)))

            # Non-dominated sorting on combined population
            fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_pop = []
            new_F = []
            new_indices = []
            for front in fronts:
                if len(new_pop) + len(front) <= n_offspring:
                    new_pop.extend([combined[i] for i in front])
                    new_F.extend([combined_F[i] for i in front])
                    new_indices.extend(front)
                else:
                    # Need to truncate this front
                    remaining = n_offspring - len(new_pop)
                    cd = crowding_distance_assignment(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    selected = sorted_front[:remaining]
                    new_pop.extend([combined[i] for i in selected])
                    new_F.extend([combined_F[i] for i in selected])
                    new_indices.extend(selected)
                    break

            if not new_pop:
                break

            current_pop = np.array(new_pop)
            current_F = np.array(new_F)
            current_indices = new_indices

        # Return the archive (non-dominated points from final population)
        # Collect all evaluated points that are non-dominated
        # For simplicity, return the final population's non-dominated points
        fronts = fast_non_dominated_sort(current_F)
        archive = []
        for i in range(len(current_F)):
            archive.append((current_pop[i], current_F[i]))

        return archive
