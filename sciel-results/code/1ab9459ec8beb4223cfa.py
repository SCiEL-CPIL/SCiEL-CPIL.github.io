"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 with hybrid global-local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Smaller population for more generations within tight budget
        pop_size = 12
        used = 0

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sort returning list of fronts."""
            n = len(F_arr)
            if n == 0:
                return []
            
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if domination_count[i] == 0]
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            remaining = set(range(n)) - set(i for front in fronts for i in front)
            if remaining:
                fronts.append(list(remaining))
            
            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            for m in range(n_obj):
                values = F_arr[indices, m]
                mn, mx = values.min(), values.max()
                if mx - mn < 1e-12:
                    cd[:] = np.inf
                    continue
                order = np.argsort(values)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, len(order) - 1):
                    cd[order[k]] += (values[order[k + 1]] - values[order[k - 1]]) / (mx - mn)
            return cd

        def local_search(x0, f0, problem, lo, hi, range_, budget_local, rng):
            """Nelder-Mead local search on the first objective (or combined)."""
            if budget_local <= 0:
                return x0, f0
            
            # Create simplex
            n = len(x0)
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                step = 0.05 * range_[i] if range_[i] > 0 else 0.05
                xi[i] += step
                xi = np.clip(xi, lo, hi)
                simplex.append(xi)
            
            # Evaluate initial simplex
            f_values = [problem(x) for x in simplex]
            local_used = n + 1
            
            # Use a combined scalar objective for local search
            # We'll minimize the sum of objectives (simple but effective)
            def scalar_obj(x):
                f = problem(x)
                return np.sum(f)
            
            # Nelder-Mead
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_iter = budget_local - local_used
            
            for _ in range(max(0, max_iter)):
                if time.time() > deadline:
                    break
                
                # Sort by scalar objective
                order = np.argsort([np.sum(f) for f in f_values])
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, lo, hi)
                f_r = problem(x_r)
                local_used += 1
                
                if np.sum(f_r) < np.sum(f_values[0]):
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = problem(x_e)
                    local_used += 1
                    if np.sum(f_e) < np.sum(f_r):
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif np.sum(f_r) < np.sum(f_values[-2]):
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = problem(x_c)
                    local_used += 1
                    if np.sum(f_c) < np.sum(f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            f_values[i] = problem(simplex[i])
                            local_used += 1
                        if local_used > budget:
                            break
                
                if local_used >= budget:
                    break
            
            # Return best point
            best_idx = np.argmin([np.sum(f) for f in f_values])
            return simplex[best_idx], f_values[best_idx]

        # Phase 1: Initialize population
        X = lo + range_ * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Phase 2: Evolutionary loop with smaller population
        while used < budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)
            
            # Select parent using tournament
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Compute crowding distance for selection
            crowd = np.zeros(n)
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, np.array(front))
                    for k, idx in enumerate(front):
                        crowd[idx] = cd[k]
            
            # Tournament selection
            i1 = rng.integers(n)
            i2 = rng.integers(n)
            if ranks[i1] < ranks[i2]:
                parent_idx = i1
            elif ranks[i2] < ranks[i1]:
                parent_idx = i2
            else:
                parent_idx = i1 if crowd[i1] > crowd[i2] else i2
            
            parent = X[parent_idx].copy()
            
            # Adaptive mutation: start larger, shrink over time
            progress = used / budget
            mut_scale = 0.08 * (1.0 - 0.7 * progress)
            mutation = mut_scale * range_ * rng.standard_normal(dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point
            max_rank = ranks.max()
            worst_candidates = np.where(ranks == max_rank)[0]
            if len(worst_candidates) > 1:
                cd = crowding_distance(F, worst_candidates)
                worst_idx = worst_candidates[np.argmin(cd)]
            else:
                worst_idx = worst_candidates[0]
            
            X[worst_idx] = child
            F[worst_idx] = fc

        # Phase 3: Local search on non-dominated points
        if used < budget:
            # Get non-dominated points
            fronts = fast_non_dominated_sort(F)
            nd_indices = fronts[0] if len(fronts) > 0 else []
            
            # Allocate remaining budget to local search on top points
            remaining = budget - used
            n_nd = len(nd_indices)
            
            if n_nd > 0 and remaining > 0:
                # Focus on the most promising (best crowding distance)
                cd = crowding_distance(F, np.array(nd_indices))
                # Sort by crowding distance descending (most diverse first)
                order = np.argsort(-cd)
                order = order[:min(n_nd, 5)]  # Local search on top 5
                
                budget_per_ls = max(1, remaining // len(order))
                
                for idx in order:
                    if used >= budget or time.time() > deadline:
                        break
                    x_best, f_best = local_search(
                        X[idx], F[idx], problem, lo, hi, range_, 
                        min(budget_per_ls, budget - used), rng
                    )
                    # Update if improved
                    if np.all(f_best <= F[idx]) and np.any(f_best < F[idx]):
                        X[idx] = x_best
                        F[idx] = f_best

        return X
# EVOLVE-BLOCK-END
