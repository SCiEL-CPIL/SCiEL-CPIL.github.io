"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """NSGA-II multi-objective optimizer with elitism, crowding distance, and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # i dominates j?
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1

        # First front: solutions not dominated by any
        front = [i for i in range(n) if n_dom[i] == 0]
        while front:
            fronts.append(front)
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            front = next_front

        # Any remaining unassigned (shouldn't happen but safety)
        assigned = set()
        for f in fronts:
            assigned.update(f)
        if len(assigned) < n:
            fronts.append([i for i in range(n) if i not in assigned])

        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices in F."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        F_sub = F[indices]
        n_obj = F_sub.shape[1]

        for m in range(n_obj):
            vals = F_sub[:, m]
            min_v, max_v = vals.min(), vals.max()
            if max_v - min_v < 1e-15:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for k in range(1, n - 1):
                dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_v - min_v)

        return dist

    def _select_parents(self, X, F, pop_size):
        """Tournament selection based on rank and crowding distance."""
        fronts = self._non_dominated_sort(F)
        ranks = np.zeros(len(F), dtype=int)
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank

        # Crowding distance per front
        cd = np.zeros(len(F))
        for front in fronts:
            cd[front] = self._crowding_distance(F, front)

        # Tournament: pick 2 random, choose better by (rank, cd)
        parents = []
        for _ in range(pop_size):
            i1 = np.random.randint(len(F))
            i2 = np.random.randint(len(F))
            if ranks[i1] < ranks[i2]:
                parents.append(i1)
            elif ranks[i2] < ranks[i1]:
                parents.append(i2)
            else:
                parents.append(i1 if cd[i1] > cd[i2] else i2)
        return parents

    def _sbx_crossover(self, p1, p2, lo, hi, eta=20.0):
        """Simulated Binary Crossover."""
        child1, child2 = p1.copy(), p2.copy()
        for j in range(self.dim):
            if np.random.random() < 0.5:
                if abs(p1[j] - p2[j]) > 1e-15:
                    if np.random.random() < 0.5:
                        u = np.random.random()
                        beta = (2 * u) ** (1.0 / (eta + 1)) if u <= 0.5 else (2 * (1 - u)) ** (1.0 / (eta + 1))
                        c1 = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
                        c2 = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
                        child1[j] = np.clip(c1, lo[j], hi[j])
                        child2[j] = np.clip(c2, lo[j], hi[j])
                    else:
                        u = np.random.random()
                        beta = (2 * u) ** (1.0 / (eta + 1)) if u <= 0.5 else (2 * (1 - u)) ** (1.0 / (eta + 1))
                        c1 = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
                        c2 = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
                        child1[j] = np.clip(c1, lo[j], hi[j])
                        child2[j] = np.clip(c2, lo[j], hi[j])
        return child1, child2

    def _polynomial_mutation(self, x, lo, hi, eta=20.0, prob=1.0 / self.dim):
        """Polynomial mutation."""
        x = x.copy()
        for j in range(self.dim):
            if np.random.random() < prob:
                delta1 = (x[j] - lo[j]) / (hi[j] - lo[j] + 1e-15)
                delta2 = (hi[j] - x[j]) / (hi[j] - lo[j] + 1e-15)
                u = np.random.random()
                if u < 0.5:
                    xy = 1 - delta1
                    val = 2 * u + (1 - 2 * u) * xy ** (eta + 1)
                    deltaq = val ** (1.0 / (eta + 1)) - 1
                else:
                    xy = 1 - delta2
                    val = 2 * (1 - u) + 2 * (u - 0.5) * xy ** (eta + 1)
                    deltaq = 1 - val ** (1.0 / (eta + 1))
                x[j] += deltaq * (hi[j] - lo[j])
                x[j] = np.clip(x[j], lo[j], hi[j])
        return x

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Initial population
        pop_size = min(60, budget)
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main NSGA-II loop
        while used < budget:
            if time.time() > deadline - 60:
                break

            # Selection
            parent_indices = self._select_parents(X, F, pop_size)

            # Crossover + Mutation to create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, budget - used)

            for k in range(0, n_offspring, 2):
                p1 = X[parent_indices[k]]
                p2 = X[parent_indices[k + 1]]
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                if used >= budget:
                    break
                if time.time() > deadline - 60:
                    break
                offspring_X.append(c2)
                offspring_F.append(problem(c2))
                used += 1

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort
            fronts = self._non_dominated_sort(combined_F)

            # Select next generation
            next_X = []
            next_F = []
            total = 0
            for front in fronts:
                if total + len(front) <= pop_size:
                    next_X.append(combined_X[front])
                    next_F.append(combined_F[front])
                    total += len(front)
                else:
                    remaining = pop_size - total
                    cd = self._crowding_distance(combined_F, front)
                    order = np.argsort(-cd)  # descending crowding distance
                    selected = [front[i] for i in order[:remaining]]
                    next_X.append(combined_X[selected])
                    next_F.append(combined_F[selected])
                    total += remaining
                    break

            X = np.vstack(next_X)
            F = np.vstack(next_F)

        # Final local polish on the non-dominated front
        if time.time() < deadline - 30:
            fronts = self._non_dominated_sort(F)
            if fronts:
                front_indices = fronts[0]
                for idx in front_indices:
                    if time.time() > deadline - 10:
                        break
                    x_best = X[idx].copy()
                    f_best = F[idx].copy()

                    # Try to improve each objective individually
                    for obj_idx in range(self.n_obj):
                        if time.time() > deadline - 10:
                            break

                        def objective_func(x):
                            try:
                                f = problem(x)
                                return f[obj_idx]
                            except:
                                return 1e10

                        # Local search around current best
                        best_x = x_best.copy()
                        best_f = f_best.copy()

                        # Use Nelder-Mead for local refinement
                        try:
                            res = minimize(
                                objective_func,
                                best_x,
                                method='Nelder-Mead',
                                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                            )
                            if res.success and res.fun < best_f[obj_idx]:
                                # Check if still better in all objectives
                                new_f = problem(res.x)
                                if np.all(new_f <= best_f + 1e-10):
                                    best_x = res.x
                                    best_f = new_f
                        except:
                            pass

                        # Update if improved
                        if np.any(best_f < f_best):
                            # Check dominance
                            if np.all(best_f <= f_best) and np.any(best_f < f_best):
                                X[idx] = best_x
                                F[idx] = best_f
                                f_best = best_f

                    # Add polished point to population if it's non-dominated
                    if len(fronts) > 0:
                        pass  # Already in population

        return X
# EVOLVE-BLOCK-END
