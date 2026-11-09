"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = min(40, max(10, self.budget // 5))

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Return True if a dominates b (a is better in all objectives and at least one strictly)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Return list of fronts, each front is a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            nd = [0] * n  # number of solutions dominating i
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif dominates(F[j], F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)

            i = 0
            while len(fronts[i]) > 0:
                next_front = []
                for k in fronts[i]:
                    for m in S[k]:
                        nd[m] -= 1
                        if nd[m] == 0:
                            next_front.append(m)
                i += 1
                if next_front:
                    fronts.append(next_front)
            return fronts[:-1] if len(fronts) > 1 else fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for indices in front_indices."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            cd = [0.0] * n
            for m in range(n_obj):
                vals = [F[idx, m] for idx in front_indices]
                sorted_order = np.argsort(vals)
                cd[sorted_order[0]] = float('inf')
                cd[sorted_order[-1]] = float('inf')
                val_range = vals[sorted_order[-1]] - vals[sorted_order[0]]
                if val_range == 0:
                    continue
                for k in range(1, n - 1):
                    cd[sorted_order[k]] += (vals[sorted_order[k + 1]] - vals[sorted_order[k - 1]]) / val_range
            return cd

        # Main evolutionary loop
        while used < self.budget:
            # Check time
            if time.time() > deadline - 2.0:
                break

            # Sort population into non-dominated fronts
            fronts = fast_non_dominated_sort(F)

            # Create offspring
            # Select parents using tournament selection based on rank and crowding distance
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            # Compute crowding distances per front
            cd = np.zeros(len(F))
            for front in fronts:
                cds = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    cd[idx] = cds[i]

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, self.budget - used)

            for _ in range(n_offspring):
                # Check time
                if time.time() > deadline - 2.0:
                    break

                # Tournament selection: pick 2 random parents, choose the better one
                i1, i2 = rng.integers(0, len(F), size=2)
                if rank[i1] < rank[i2]:
                    parent = i1
                elif rank[i2] < rank[i1]:
                    parent = i2
                else:
                    parent = i1 if cd[i1] > cd[i2] else i2

                # Mutation: Gaussian perturbation
                child = X[parent].copy()
                sigma = 0.1 * (hi - lo)
                child += sigma * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

                if used >= self.budget:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Select next generation
            combined_fronts = fast_non_dominated_sort(combined_F)

            next_X = []
            next_F = []
            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(next_X)
                    cds = crowding_distance(combined_F, front)
                    sorted_order = np.argsort(cds)[::-1]  # descending crowding distance
                    for k in range(remaining):
                        idx = front[sorted_order[k]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            if next_X:
                X = np.array(next_X)
                F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
