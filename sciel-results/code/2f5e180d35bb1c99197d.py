"""Optimized multi-objective search for DTLZ4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer with deadline guard for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 2))
        # Ensure we don't exceed budget
        if pop_size > budget:
            pop_size = budget

        # Initialize population
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper: dominance check (vectorized)
        def dominates(a, b):
            """Return True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F):
            """Return indices of non-dominated solutions in F."""
            n_sol = len(F)
            nd_idx = []
            for i in range(n_sol):
                is_nd = True
                for j in range(n_sol):
                    if i == j:
                        continue
                    if dominates(F[j], F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd_idx.append(i)
            return nd_idx

        # Fast non-dominated sorting for crowding distance
        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n_sol = len(F)
            S = [[] for _ in range(n_sol)]  # S[i] = indices dominated by i
            n_dominated = [0] * n_sol       # number of solutions dominated by i
            fronts = [[]]

            for i in range(n_sol):
                for j in range(i + 1, n_sol):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dominated[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_dominated[i] += 1
                if n_dominated[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)

            # Remove empty trailing fronts
            while fronts and not fronts[-1]:
                fronts.pop()
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return [float('inf')] * len(indices)

            dist = [0.0] * len(indices)
            F_sub = F[indices]
            n_obj = F_sub.shape[1]

            for k in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, k])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                if F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k] == 0:
                    continue
                for i in range(1, len(sorted_idx) - 1):
                    dist[sorted_idx[i]] += (
                        F_sub[sorted_idx[i + 1], k] - F_sub[sorted_idx[i - 1], k]
                    ) / (F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k])

            return dist

        # Adaptive mutation rate
        base_mut_rate = 0.5
        base_sigma = 0.2

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Select parents: pick from first front(s)
            # Use tournament-like selection from non-dominated fronts
            parent_pool = fronts[0] if fronts[0] else list(range(len(F)))
            if len(parent_pool) < 2:
                parent_pool = list(range(len(F)))

            # Select two parents
            p1_idx = parent_pool[rng.integers(len(parent_pool))]
            p2_idx = parent_pool[rng.integers(len(parent_pool))]
            while p2_idx == p1_idx and len(parent_pool) > 1:
                p2_idx = parent_pool[rng.integers(len(parent_pool))]

            # SBX crossover
            if rng.random() < 0.9:
                # Simulated Binary Crossover
                eta_c = 20.0
                child1 = np.zeros(n)
                child2 = np.zeros(n)
                for i in range(n):
                    x1 = X[p1_idx, i]
                    x2 = X[p2_idx, i]
                    if x1 == x2:
                        child1[i] = x1
                        child2[i] = x2
                        continue
                    if x1 > x2:
                        x1, x2 = x2, x1
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    child1[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    child2[i] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
            else:
                child1 = X[p1_idx].copy()
                child2 = X[p2_idx].copy()

            # Polynomial mutation
            eta_m = 20.0
            for i in range(n):
                if rng.random() < base_mut_rate:
                    delta1 = (2 * rng.random()) ** (1.0 / (eta_m + 1)) - 1
                    delta2 = (1 - 2 * rng.random()) ** (1.0 / (eta_m + 1))
                    span = hi[i] - lo[i]
                    child1[i] += span * (delta1 * base_sigma)
                    child2[i] += span * (delta2 * base_sigma)

            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)

            # Evaluate children
            f1 = problem(child1)
            used += 1
            if used >= budget:
                # Only add if we have budget
                # Replace worst solution
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                X[worst_idx] = child1
                F[worst_idx] = f1
                break

            f2 = problem(child2)
            used += 1

            # Add to population and remove worst
            # Simple strategy: replace the two worst solutions
            # Find worst based on front rank and crowding
            # For simplicity, replace worst by sum of objectives
            worst1 = int(np.argmax(np.sum(F, axis=1)))
            F[worst1] = f1
            X[worst1] = child1

            worst2 = int(np.argmax(np.sum(F, axis=1)))
            F[worst2] = f2
            X[worst2] = child2

        return X
# EVOLVE-BLOCK-END
