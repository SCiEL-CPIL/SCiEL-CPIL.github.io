"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(40, max(10, budget // 5))

        # Random seed for reproducibility within the call
        rng = np.random.default_rng(42)

        def dominate(a, b):
            """Check if a dominates b (a is better than or equal in all, strictly better in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Perform fast non-dominated sorting. Returns list of fronts (lists of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of solutions dominated by i
            n_dom = [0] * n  # n_dom[i] = number of solutions that dominate i
            fronts = [[]]

            for i in range(n):
                for j in range(i + 1, n):
                    if dominate(F[i], F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif dominate(F[j], F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

                if n_dom[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)

            # Remove empty last front if any
            if fronts and not fronts[-1]:
                fronts.pop()

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n

            cd = [0.0] * n
            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = sorted(range(n), key=lambda i: F[front_indices[i]][m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')

                f_min = F[front_indices[sorted_idx[0]]][m]
                f_max = F[front_indices[sorted_idx[-1]]][m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue

                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (
                        F[front_indices[sorted_idx[i + 1]]][m] -
                        F[front_indices[sorted_idx[i - 1]]][m]
                    ) / f_range

            return cd

        def tournament_select(F, fronts, crowd_dists, k=2):
            """Tournament selection based on rank and crowding distance."""
            candidates = rng.integers(0, len(F), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                # Compare rank
                rank_best = fronts.index(next(f for f in fronts if best in f)) if best in [i for f in fronts for i in f] else 0
                rank_c = fronts.index(next(f for f in fronts if c in f)) if c in [i for f in fronts for i in f] else 0

                # More efficient: precompute ranks
                # We'll use a different approach
                pass

            # Simpler: just pick the one with better rank, then crowding
            # Precompute rank for each index
            rank = [0] * len(F)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best]:
                    best = c
                elif rank[c] == rank[best]:
                    # Find crowding distances
                    cd_best = crowd_dists.get(best, 0)
                    cd_c = crowd_dists.get(c, 0)
                    if cd_c > cd_best:
                        best = c
            return best

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Main loop
        while used < budget:
            # Evaluate current population
            fronts = fast_non_dominated_sort(F)

            # Compute crowding distances for each front
            crowd_dists = {}
            for front in fronts:
                cd = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    crowd_dists[idx] = cd[i]

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, budget - used)

            for _ in range(n_offspring):
                # Select parents
                p1 = tournament_select(F, fronts, crowd_dists)
                p2 = tournament_select(F, fronts, crowd_dists)

                # SBX crossover
                child = X[p1].copy()
                if rng.random() < 0.9:  # Crossover probability
                    for j in range(n_dim):
                        if rng.random() < 0.5:
                            # SBX
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (10 + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (10 + 1))
                            child[j] = 0.5 * ((1 + beta) * X[p1][j] + (1 - beta) * X[p2][j])

                # Mutation
                if rng.random() < 0.9:  # Mutation probability
                    for j in range(n_dim):
                        if rng.random() < 1.0 / n_dim:
                            # Polynomial mutation
                            u = rng.random()
                            if u < 0.5:
                                delta = (2 * u) ** (1 / (20 + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1 / (20 + 1))
                            child[j] += delta * (hi[j] - lo[j])

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

                if used >= budget:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Selection
            combined_fronts = fast_non_dominated_sort(combined_F)
            new_F = []
            new_X = []
            for front in combined_fronts:
                if len(new_F) + len(front) <= pop_size:
                    for idx in front:
                        new_F.append(combined_F[idx])
                        new_X.append(combined_X[idx])
                else:
                    # Need to select some from this front
                    remaining = pop_size - len(new_F)
                    # Sort by crowding distance
                    cd = crowding_distance(combined_F, front)
                    sorted_by_cd = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in range(remaining):
                        idx = front[sorted_by_cd[i]]
                        new_F.append(combined_F[idx])
                        new_X.append(combined_X[idx])
                    break

            if not new_X:
                break

            X = np.array(new_X)
            F = np.array(new_F)

        return X

# EVOLVE-BLOCK-END
