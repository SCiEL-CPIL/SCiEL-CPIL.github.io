import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using NSGA-II with Sobol initialization and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """
        Perform fast non-dominated sorting.
        Returns a list of fronts, where each front is a list of indices.
        """
        n = len(F)
        if n == 0:
            return []

        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dominating = [0] * n  # n_dominating[i] = number of solutions dominating i
        rank = [0] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dominating[i] += 1

        front1 = []
        for i in range(n):
            if n_dominating[i] == 0:
                rank[i] = 0
                front1.append(i)

        fronts = [front1]
        k = 1
        while front1:
            next_front = []
            for i in front1:
                for j in S[i]:
                    n_dominating[j] -= 1
                    if n_dominating[j] == 0:
                        rank[j] = k
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
                k += 1
            front1 = next_front

        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for a set of indices in F.
        Returns an array of distances for the given indices.
        """
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        distances = np.zeros(n)
        F_sub = F[indices]
        n_obj = F_sub.shape[1]

        for m in range(n_obj):
            # Sort by objective m
            order = np.argsort(F_sub[:, m])
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf

            f_min = F_sub[order[0], m]
            f_max = F_sub[order[-1], m]
            if f_max - f_min > 1e-15:
                for i in range(1, n - 1):
                    distances[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / (f_max - f_min)

        return distances

    def _tournament_select(self, population_F, ranks, crowding, k=2):
        """
        Tournament selection. Returns index of winner in population.
        """
        n = len(population_F)
        candidates = np.random.choice(n, size=k, replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            # Prefer lower rank (better front)
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best]:
                # Prefer higher crowding distance
                if crowding[c] > crowding[best]:
                    best = c
        return best

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size: use a reasonable fraction of budget
        pop_size = min(50, self.budget // 4)
        if pop_size < 10:
            pop_size = max(10, self.budget // 5)

        # Initialize population with Sobol points for good spread
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        X_init = lo + sobol.random(pop_size) * range_

        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        eval_count = pop_size

        # Keep track of all evaluated points for final result
        all_X = list(X_init)
        all_F = list(F_init)

        # NSGA-II main loop
        population_X = X_init.copy()
        population_F = F_init.copy()

        while eval_count < self.budget:
            if time.time() > deadline - 10:
                break

            # Compute ranks and crowding distance
            fronts = self._fast_non_dominated_sort(population_F)
            n_pop = len(population_F)
            ranks = np.zeros(n_pop, dtype=int)
            for rank_idx, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank_idx

            # Compute crowding distance per front
            crowding = np.zeros(n_pop)
            for front in fronts:
                if len(front) > 0:
                    cd = self._crowding_distance(population_F, front)
                    for i, idx in enumerate(front):
                        crowding[idx] = cd[i]

            # Create offspring
            offspring_size = pop_size
            offspring_X = []
            offspring_F = []

            for _ in range(offspring_size):
                if eval_count + 1 > self.budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Tournament selection for two parents
                p1_idx = self._tournament_select(population_F, ranks, crowding)
                p2_idx = self._tournament_select(population_F, ranks, crowding)

                p1 = population_X[p1_idx]
                p2 = population_X[p2_idx]

                # Simulated binary crossover (SBX)
                if np.random.random() < 0.9:  # Crossover probability
                    beta = 15  # Distribution index
                    child1 = np.zeros(self.dim)
                    child2 = np.zeros(self.dim)
                    for d in range(self.dim):
                        if np.random.random() < 0.5:
                            u = np.random.random()
                            if u <= 1 / (beta + 1):
                                alpha = (2 * u) ** (1 / (beta + 1))
                            elif u >= 1 - 1 / (beta + 1):
                                alpha = (2 * (1 - u)) ** (1 / (beta + 1))
                            else:
                                alpha = 1.0

                            if p1[d] < p2[d]:
                                child1[d] = 0.5 * ((1 + alpha) * p1[d] + (1 - alpha) * p2[d])
                                child2[d] = 0.5 * ((1 - alpha) * p1[d] + (1 + alpha) * p2[d])
                            else:
                                child1[d] = 0.5 * ((1 + alpha) * p2[d] + (1 - alpha) * p1[d])
                                child2[d] = 0.5 * ((1 - alpha) * p2[d] + (1 + alpha) * p1[d])
                        else:
                            child1[d] = p1[d]
                            child2[d] = p2[d]
                else:
                    child1 = p1.copy()
                    child2 = p2.copy()

                # Polynomial mutation
                eta = 20  # Distribution index
                for d in range(self.dim):
                    if np.random.random() < 0.5 / (self.dim + 1):  # Mutation probability
                        u = np.random.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / (eta + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                        # Scale mutation by the range
                        child1[d] += delta * (hi[d] - lo[d])

                    if np.random.random() < 0.5 / (self.dim + 1):
                        u = np.random.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / (eta + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                        child2[d] += delta * (hi[d] - lo[d])

                # Clip to bounds
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                # Evaluate
                f1 = problem(child1)
                eval_count += 1
                all_X.append(child1)
                all_F.append(f1)
                offspring_X.append(child1)
                offspring_F.append(f1)

                if eval_count >= self.budget:
                    break

                f2 = problem(child2)
                eval_count += 1
                all_X.append(child2)
                all_F.append(f2)
                offspring_X.append(child2)
                offspring_F.append(f2)

                if eval_count >= self.budget:
                    break

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F)

            # Combine parent and offspring populations
            combined_X = np.vstack([population_X, np.array(offspring_X)])
            combined_F = np.vstack([population_F, offspring_F])

            # Fast non-dominated sort on combined population
            fronts = self._fast_non_dominated_sort(combined_F)

            # Select next population
            new_population_X = []
            new_population_F = []
            n_combined = len(combined_F)
            ranks = np.zeros(n_combined, dtype=int)
            for rank_idx, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank_idx

            for front in fronts:
                if len(new_population_X) + len(front) <= pop_size:
                    # Take entire front
                    for idx in front:
                        new_population_X.append(combined_X[idx])
                        new_population_F.append(combined_F[idx])
                else:
                    # Need to select some from this front based on crowding distance
                    remaining = pop_size - len(new_population_X)
                    cd = self._crowding_distance(combined_F, front)
                    # Sort by crowding distance (descending)
                    order = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[order[i]]
                        new_population_X.append(combined_X[idx])
                        new_population_F.append(combined_F[idx])
                    break

            if not new_population_X:
                break

            population_X = np.array(new_population_X)
            population_F = np.array(new_population_F)

        # Local refinement of non-dominated points
        if eval_count < self.budget and time.time() < deadline - 15:
            # Find non-dominated points from all evaluated points
            all_F_arr = np.array(all_F)
            fronts_final = self._fast_non_dominated_sort(all_F_arr)
            if fronts_final:
                nd_indices = fronts_final[0]
                # Select a few non-dominated points for local refinement
                n_refine = min(5, len(nd_indices))
                if n_refine > 0:
                    # Pick diverse points from the non-dominated front
                    if len(nd_indices) > n_refine:
                        # Use crowding distance to select diverse points
                        cd = self._crowding_distance(all_F_arr, nd_indices)
                        order = np.argsort(-cd)
                        selected = [nd_indices[i] for i in order[:n_refine]]
                    else:
                        selected = nd_indices

                    for idx in selected:
                        if eval_count >= self.budget or time.time() > deadline - 5:
                            break
                        x0 = all_X[idx]
                        # Local Gaussian search
                        n_local = min(10, self.budget - eval_count)
                        for _ in range(n_local):
                            if eval_count >= self.budget or time.time() > deadline - 5:
                                break
                            sigma = 0.05
                            x_new = x0 + np.random.normal(0, sigma, self.dim)
                            x_new = np.clip(x_new, lo, hi)
                            f_new = problem(x_new)
                            eval_count += 1
                            all_X.append(x_new)
                            all_F.append(f_new)

        return np.array(all_X)
