"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining LHS initialization, NSGA-II, and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Count evaluations
        eval_count = 0

        def eval_point(x):
            nonlocal eval_count
            eval_count += 1
            return np.array(problem(x))

        # ---- Phase 1: LHS Initialization ----
        n_init = min(40, budget)
        if n_init <= 0:
            return None

        # Latin Hypercube Sampling
        indices = np.zeros((n_init, dim), dtype=int)
        for d in range(dim):
            perm = np.random.permutation(n_init)
            indices[:, d] = perm
        X_lhs = (indices + np.random.random((n_init, dim))) / n_init
        X_lhs = lo + (hi - lo) * X_lhs

        F = np.array([eval_point(x) for x in X_lhs])

        # ---- Phase 2: NSGA-II-like Evolution ----
        # Population size for evolution
        n_pop = min(30, max(10, budget // 8))
        if n_pop > budget - n_init:
            n_pop = max(5, budget - n_init)

        # Ensure we have enough budget
        if n_init + n_pop > budget:
            n_pop = max(5, budget - n_init)

        # Sort initial population by non-domination
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1
                if n_dominates[i] == 0:
                    fronts[0].append(i)

            k = 0
            while len(fronts[k]) > 0 and k + 1 < n:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-15:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_v - min_v)
            return cd

        # Initial fronts
        fronts = fast_non_dominated_sort(F)

        # Keep track of all evaluated points for final result
        all_X = X_lhs.copy()
        all_F = F.copy()

        # Evolution loop
        gen = 0
        while eval_count < budget:
            if time.time() > deadline - 10:
                break

            # Select parents from first front using crowding distance
            if len(fronts) == 0 or len(fronts[0]) == 0:
                break

            parent_indices = fronts[0]
            cd = crowding_distance(F, parent_indices)
            # Select 2 parents with highest crowding distance
            n_parents = min(2, len(parent_indices))
            if n_parents < 2:
                # Duplicate if needed
                parent_idx_list = list(parent_indices)
                while len(parent_idx_list) < 2:
                    parent_idx_list.append(parent_idx_list[np.random.randint(len(parent_idx_list))])
            else:
                top_cd = np.argsort(cd)[::-1][:n_parents]
                parent_idx_list = [parent_indices[i] for i in top_cd]

            p1_idx = parent_idx_list[0]
            p2_idx = parent_idx_list[1]

            # Crossover and mutation
            child = 0.5 * (F[p1_idx] + F[p2_idx])  # placeholder, we work in decision space
            # Actually work in decision space
            x1 = all_X[p1_idx] if p1_idx < len(all_X) else X_lhs[p1_idx % len(X_lhs)]
            x2 = all_X[p2_idx] if p2_idx < len(all_X) else X_lhs[p2_idx % len(X_lhs)]

            # Simulated binary crossover
            child_x = np.zeros(dim)
            for d in range(dim):
                u = np.random.random()
                if u < 0.5:
                    beta = (2 * u) ** (1 / 3)
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / 3)
                child_x[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])

            # Mutation
            for d in range(dim):
                if np.random.random() < 0.5:
                    child_x[d] += 0.1 * (hi[d] - lo[d]) * np.random.standard_normal()
            child_x = np.clip(child_x, lo, hi)

            if eval_count >= budget:
                break

            fc = eval_point(child_x)

            # Add to population
            all_X = np.vstack([all_X, child_x.reshape(1, -1)])
            all_F = np.vstack([all_F, fc.reshape(1, -1)])

            # Update fronts
            fronts = fast_non_dominated_sort(all_F)

            # Remove worst points to maintain population size
            max_pop = min(50, budget - eval_count + 50)
            while len(all_X) > max_pop:
                # Find last front
                last_front = fronts[-1]
                if len(last_front) > 1:
                    cd = crowding_distance(all_F, last_front)
                    worst_in_front = np.argmin(cd)
                    remove_idx = last_front[worst_in_front]
                else:
                    remove_idx = last_front[0]
                # Remove from arrays
                mask = np.ones(len(all_X), dtype=bool)
                mask[remove_idx] = False
                all_X = all_X[mask]
                all_F = all_F[mask]
                fronts = fast_non_dominated_sort(all_F)

            gen += 1

        # ---- Phase 3: Local Refinement around Non-Dominated Points ----
        remaining_budget = budget - eval_count
        if remaining_budget > 0 and time.time() < deadline - 5:
            # Get non-dominated points
            fronts = fast_non_dominated_sort(all_F)
            nd_indices = fronts[0] if len(fronts) > 0 else []

            # Try local refinement around each non-dominated point
            n_refine = min(len(nd_indices), remaining_budget // 2)
            for idx in nd_indices[:n_refine]:
                if eval_count >= budget or time.time() > deadline - 5:
                    break

                x_base = all_X[idx]
                # Try a few local perturbations
                for attempt in range(2):
                    if eval_count >= budget or time.time() > deadline - 5:
                        break
                    # Random direction
                    direction = np.random.standard_normal(dim)
                    direction /= (np.linalg.norm(direction) + 1e-15)
                    step_size = 0.05 * (hi - lo)
                    x_new = x_base + step_size * direction
                    x_new = np.clip(x_new, lo, hi)

                    fc = eval_point(x_new)
                    all_X = np.vstack([all_X, x_new.reshape(1, -1)])
                    all_F = np.vstack([all_F, fc.reshape(1, -1)])

        # Final: get non-dominated set
        fronts = fast_non_dominated_sort(all_F)
        nd_indices = fronts[0] if len(fronts) > 0 else range(len(all_F))
        best_X = all_X[nd_indices]
        best_F = all_F[nd_indices]

        return best_X
# EVOLVE-BLOCK-END
