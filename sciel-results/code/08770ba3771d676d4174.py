"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II style with crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a reasonable fraction of budget
        pop_size = min(40, max(10, budget // 5))

        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        def fast_non_dominated_sort(F):
            """Return list of fronts, each front is a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            fronts = [[]]

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        nd[p] += 1
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        nd[q] += 1
                        if nd[q] == 0:
                            fronts[-1].append(q)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts[:-1] if len(fronts) > 1 else fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n

            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                min_val = vals.min()
                max_val = vals.max()
                if max_val - min_val < 1e-12:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = float('inf')
                cd[order[-1]] = float('inf')
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_val - min_val)
            return cd

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size and used + len(offspring_X) < budget:
                # Binary tournament selection
                i, j = rng.integers(0, pop_size, size=2)
                if F[i, 0] < F[j, 0] or (F[i, 0] == F[j, 0] and rng.random() < 0.5):
                    p1 = X[i]
                else:
                    p1 = X[j]

                k, l = rng.integers(0, pop_size, size=2)
                if F[k, 0] < F[l, 0] or (F[k, 0] == F[l, 0] and rng.random() < 0.5):
                    p2 = X[k]
                else:
                    p2 = X[l]

                # SBX crossover
                if rng.random() < 0.9:
                    eta = 20
                    child = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (eta + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                            child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                        else:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (eta + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                            child[d] = 0.5 * ((1 + beta) * p2[d] + (1 - beta) * p1[d])
                else:
                    child = p1.copy()

                # Polynomial mutation
                eta_mut = 20
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u <= 0.5:
                            delta = (2 * u) ** (1 / (eta_mut + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (eta_mut + 1))
                        child[d] += delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            all_X.append(offspring_X.copy())
            all_F.append(offspring_F.copy())

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Fast non-dominated sort
            fronts = fast_non_dominated_sort(combined_F)

            # Build new population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Sort by crowding distance and take remaining
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(-cd)
                    remaining = pop_size - len(new_X)
                    for idx in order[:remaining]:
                        new_X.append(combined_X[front[idx]])
                        new_F.append(combined_F[front[idx]])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        # Return the best non-dominated points from all evaluations
        all_X = np.vstack(all_X)
        all_F = np.vstack(all_F)

        # Filter to only points strictly better than reference point [3.0, 5.0, 7.0]
        ref = np.array([3.0, 5.0, 7.0])
        mask = np.all(all_F < ref, axis=1)
        if np.any(mask):
            best_X = all_X[mask]
            best_F = all_F[mask]
        else:
            best_X = all_X
            best_F = all_F

        # Return all evaluated points (the evaluator records all objective vectors)
        return all_X
# EVOLVE-BLOCK-END
