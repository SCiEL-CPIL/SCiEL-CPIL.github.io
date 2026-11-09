"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid MOEA: DE-based offspring with NSGA-II style selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = 20

        # Initialize population
        rng = np.random.default_rng(0)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Non-dominated sort (vectorized for speed)
        def non_dominated_sort(F_mat):
            n = len(F_mat)
            S = [[] for _ in range(n)]
            n_doms = [0] * n
            ranks = [0] * n
            current_front = []

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    # p dominates q?
                    if np.all(F_mat[p] <= F_mat[q]) and np.any(F_mat[p] < F_mat[q]):
                        S[p].append(q)
                    elif np.all(F_mat[q] <= F_mat[p]) and np.any(F_mat[q] < F_mat[p]):
                        n_doms[p] += 1

                if n_doms[p] == 0:
                    ranks[p] = 0
                    current_front.append(p)

            i = 0
            while current_front:
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_doms[q] -= 1
                        if n_doms[q] == 0:
                            ranks[q] = i + 1
                            next_front.append(q)
                i += 1
                current_front = next_front

            fronts = [[] for _ in range(i)]
            for idx, r in enumerate(ranks):
                if r < i:
                    fronts[r].append(idx)
            return fronts

        # Crowding distance
        def crowding_distance(F_mat, indices):
            if not indices:
                return {}
            n = len(indices)
            if n <= 2:
                return {idx: float('inf') for idx in indices}

            dists = {idx: 0.0 for idx in indices}
            m = n_obj
            for k in range(m):
                sorted_idx = sorted(indices, key=lambda i: F_mat[i][k])
                dists[sorted_idx[0]] = float('inf')
                dists[sorted_idx[-1]] = float('inf')
                f_min = F_mat[sorted_idx[0]][k]
                f_max = F_mat[sorted_idx[-1]][k]
                if f_max == f_min:
                    continue
                for j in range(1, n - 1):
                    idx = sorted_idx[j]
                    dists[idx] += (F_mat[sorted_idx[j + 1]][k] - F_mat[sorted_idx[j - 1]][k]) / (f_max - f_min)
            return dists

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Generate offspring using DE strategy
            offspring_X = []
            for i in range(pop_size):
                p1_idx = rng.integers(pop_size)
                p2_idx = rng.integers(pop_size)
                p3_idx = rng.integers(pop_size)

                p1 = X[p1_idx]
                p2 = X[p2_idx]
                p3 = X[p3_idx]

                # DE mutation
                F_cr = 0.5
                x_child = p1 + F_cr * (p2 - p3)

                # Gaussian mutation
                x_child += 0.1 * span * rng.standard_normal(dim)
                x_child = np.clip(x_child, lo, hi)

                offspring_X.append(x_child)

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size

            # Combine parents and offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])

            # Non-dominated sort
            fronts = non_dominated_sort(F_comb)

            # Select next population
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                else:
                    cd = crowding_distance(F_comb, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    needed = pop_size - len(X_new)
                    for idx in sorted_front[:needed]:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    break

            X = np.array(X_new)
            F = np.array(F_new)

        return F
# EVOLVE-BLOCK-END
