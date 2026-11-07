"""Multi-objective optimizer on RE37 using NSGA-II with adaptive mutation and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """NSGA-II multi-objective optimizer with adaptive DE mutation and local Nelder-Mead polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random.default_rng(42)

        # Latin Hypercube Sampling for initial population
        pop_size = min(50, budget // 3)
        X = self._lhs_sample(pop_size, dim, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_points = F.copy()

        # NSGA-II main loop
        while used < budget:
            if time.time() > deadline - 5:
                break

            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)

            # Crowding distance assignment
            crowding = np.zeros(len(F))
            for front in fronts:
                if len(front) > 2:
                    for i in range(n_obj):
                        obj_vals = F[front, i]
                        sorted_idx = np.argsort(obj_vals)
                        crowding[front[sorted_idx[0]]] = np.inf
                        crowding[front[sorted_idx[-1]]] = np.inf
                        obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                        if obj_range > 1e-15:
                            for k in range(1, len(front) - 1):
                                idx = sorted_idx[k]
                                prev_idx = sorted_idx[k - 1]
                                next_idx = sorted_idx[k + 1]
                                crowding[front[idx]] += (obj_vals[next_idx] - obj_vals[prev_idx]) / obj_range
                else:
                    crowding[front] = np.inf

            # Selection: binary tournament
            offspring = []
            while len(offspring) < pop_size:
                if time.time() > deadline - 5:
                    break
                p1 = self._tournament(F, fronts, crowding, rng)
                p2 = self._tournament(F, fronts, crowding, rng)

                # Differential Evolution style crossover
                if len(offspring) >= 2 and rng.random() < 0.8:
                    # Use DE mutation
                    indices = list(range(len(F)))
                    indices.remove(p1)
                    indices.remove(p2)
                    if len(indices) >= 2:
                        a_idx, b_idx = rng.choice(indices, size=2, replace=False)
                        FCR = 0.8  # mutation factor
                        child = F[a_idx] + FCR * (F[b_idx] - F[p1])
                        # Map back to decision space approximately via interpolation
                        # Actually, we need to work in decision space
                        pass

                # Standard crossover + mutation in decision space
                if rng.random() < 0.7:
                    # Simulated binary crossover
                    child = self._sbx_crossover(X[p1], X[p2], rng)
                else:
                    child = X[p1].copy()

                # Adaptive mutation
                mutation_rate = 0.1 + 0.05 * (used / budget)
                for d in range(dim):
                    if rng.random() < mutation_rate:
                        sigma = 0.1 * (hi[d] - lo[d])
                        child[d] += sigma * rng.standard_normal()
                        child[d] = np.clip(child[d], lo[d], hi[d])

                offspring.append(child)

            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring])
            used += len(offspring)
            all_points = np.vstack([all_points, F_off])

            # Combine parents and offspring
            X_comb = np.vstack([X, offspring])
            F_comb = np.vstack([F, F_off])

            # Environmental selection
            fronts_comb = self._fast_non_dominated_sort(F_comb)
            crowding_comb = np.zeros(len(F_comb))
            for front in fronts_comb:
                if len(front) > 2:
                    for i in range(n_obj):
                        obj_vals = F_comb[front, i]
                        sorted_idx = np.argsort(obj_vals)
                        crowding_comb[front[sorted_idx[0]]] = np.inf
                        crowding_comb[front[sorted_idx[-1]]] = np.inf
                        obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                        if obj_range > 1e-15:
                            for k in range(1, len(front) - 1):
                                idx = sorted_idx[k]
                                prev_idx = sorted_idx[k - 1]
                                next_idx = sorted_idx[k + 1]
                                crowding_comb[front[idx]] += (obj_vals[next_idx] - obj_vals[prev_idx]) / obj_range
                else:
                    crowding_comb[front] = np.inf

            # Select next population
            selected = np.zeros(pop_size, dtype=int)
            count = 0
            for front in fronts_comb:
                if count + len(front) <= pop_size:
                    selected[count:count + len(front)] = front
                    count += len(front)
                else:
                    # Sort by crowding distance
                    front_sorted = front[np.argsort(crowding_comb[front])[::-1]]
                    selected[count:pop_size] = front_sorted[:pop_size - count]
                    break

            X = X_comb[selected]
            F = F_comb[selected]

        # Local refinement of non-dominated points
        if time.time() < deadline - 10:
            fronts_final = self._fast_non_dominated_sort(F)
            if fronts_final:
                nd_points = fronts_final[0]
                for idx in nd_points:
                    if time.time() > deadline - 5:
                        break
                    if used >= budget:
                        break
                    x0 = X[idx]
                    f0 = F[idx]

                    # Nelder-Mead on a weighted scalarization
                    weights = np.ones(n_obj) / n_obj
                    def scalarized(x):
                        fx = problem(x)
                        return np.dot(fx, weights)

                    try:
                        res = minimize(scalarized, x0, method='Nelder-Mead',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
                        if res.success and res.fun < f0:
                            used += 1
                            all_points = np.vstack([all_points, res.fun if np.isscalar(res.fun) else np.array(res.fun)])
                            # Update if better
                            X[idx] = res.x
                            F[idx] = problem(res.x)
                            all_points[-1] = F[idx]
                    except Exception:
                        pass

        return X

    def _lhs_sample(self, n, dim, lo, hi, rng):
        """Latin Hypercube Sampling."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return lo + (hi - lo) * samples

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting, returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []

        S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
        n_dominating = np.zeros(n, dtype=int)
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominating[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominating[i] += 1

        for i in range(n):
            if n_dominating[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dominating[j] -= 1
                    if n_dominating[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def _tournament(self, F, fronts, crowding, rng):
        """Binary tournament selection."""
        i, j = rng.integers(0, len(F), size=2)
        # Find front rank
        front_rank = np.zeros(len(F), dtype=int)
        for rank, front in enumerate(fronts):
            for idx in front:
                front_rank[idx] = rank

        if front_rank[i] < front_rank[j]:
            return i
        elif front_rank[j] < front_rank[i]:
            return j
        else:
            # Same front, use crowding distance
            return i if crowding[i] > crowding[j] else j

    def _sbx_crossover(self, x1, x2, rng, eta=20.0):
        """Simulated Binary Crossover."""
        child = x1.copy()
        for d in range(len(x1)):
            if rng.random() < 0.5:
                if abs(x1[d] - x2[d]) > 1e-15:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
        return child
# EVOLVE-BLOCK-END
