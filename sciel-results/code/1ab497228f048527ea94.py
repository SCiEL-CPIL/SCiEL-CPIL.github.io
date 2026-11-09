"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using hypervolume-driven adaptive sampling
    with NSGA-II-style diversity and local polishing."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)
        ref = np.array([1.1, 1.3, 1.1])

        def fast_non_dominated_sort(F_arr):
            """Returns list of fronts, each a list of indices."""
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

            current_front = [i for i in range(n) if domination_count[i] == 0]
            fronts = []
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        def compute_rank_array(F_arr, fronts):
            """Compute rank for each index (0-based, lower is better)."""
            n = len(F_arr)
            ranks = np.zeros(n, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r
            return ranks

        def crowding_distance(F_arr, front_indices):
            """Compute crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            F_front = F_arr[front_indices]
            
            for m in range(n_obj):
                sorted_idx = np.argsort(F_front[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F_front[sorted_idx[-1], m] - F_front[sorted_idx[0], m]
                if f_range > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F_front[sorted_idx[i+1], m] - F_front[sorted_idx[i-1], m]) / f_range
            
            return cd

        def hv_marginal_contribution(F_arr, nd_indices):
            """Compute approximate marginal HV contribution for each non-dominated point.
            Uses a grid-based approximation in objective space."""
            n_nd = len(nd_indices)
            if n_nd == 0:
                return np.array([])
            
            F_nd = F_arr[nd_indices]
            # Only consider points that are strictly better than reference
            valid_mask = np.all(F_nd < ref, axis=1)
            F_valid = F_nd[valid_mask]
            n_valid = len(F_valid)
            
            if n_valid == 0:
                return np.zeros(n_nd)
            
            # Normalize to [0,1] for grid computation
            f_min = F_valid.min(axis=0)
            f_max = np.minimum(F_valid.max(axis=0), ref)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0
            F_norm = (F_valid - f_min) / f_range
            
            # Compute HV contribution using inclusion-exclusion on 3D
            # For 3 objectives, marginal contribution of point i is the volume
            # of the box [F_i, ref] minus volumes already covered
            # Simplified: use the box volume as a proxy for contribution
            contribs = np.zeros(n_nd)
            for i in range(n_nd):
                if not valid_mask[i]:
                    continue
                # Box volume from F_i to ref
                box_vol = np.prod(np.maximum(ref - F_nd[i], 0))
                # Discount by proximity to other points
                if n_valid > 1:
                    # Count how many other points are "close" in objective space
                    dists = np.linalg.norm(F_valid - F_valid[i], axis=1)
                    close_count = np.sum(dists < 0.1)
                    discount = 1.0 / (1.0 + close_count)
                    contribs[i] = box_vol * discount
                else:
                    contribs[i] = box_vol
            
            return contribs

        def select_parents(F_arr, X_arr, ranks, cd_map, n_select=2):
            """Select parents using tournament on rank + crowding distance."""
            n = len(F_arr)
            parents = []
            for _ in range(n_select):
                candidates = rng.integers(n, size=3)
                best = candidates[0]
                best_key = (ranks[best], -cd_map.get(ranks[best], {}).get(best, 0))
                for c in candidates[1:]:
                    key_c = (ranks[c], -cd_map.get(ranks[c], {}).get(c, 0))
                    if key_c < best_key:
                        best = c
                        best_key = key_c
                parents.append(best)
            return parents

        def local_polish(x, f, max_iter=20, step=0.03):
            """Polish a single point using coordinate descent."""
            best_x = x.copy()
            best_f = f.copy()
            evals_used = 0
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        trial_x = best_x.copy()
                        trial_x[d] += direction * step
                        trial_x = np.clip(trial_x, lo, hi)
                        trial_f = problem(trial_x)
                        evals_used += 1
                        if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                            best_x = trial_x
                            best_f = trial_f
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            return best_x, best_f, evals_used

        # Phase 1: Latin Hypercube Sampling initialization
        pop_size = min(40, budget)
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))

        for d in range(dim):
            perm = rng.permutation(pop_size)
            X[perm, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(pop_size) + rng.random(pop_size)) / pop_size

        used = 0
        for i in range(pop_size):
            F[i] = problem(X[i])
            used += 1
            if time.time() > deadline:
                break

        # Phase 2: NSGA-II-style evolution with HV-guided mutation
        while used < budget:
            if time.time() > deadline:
                break

            # Compute fronts, ranks, and crowding distances
            fronts = fast_non_dominated_sort(F)
            ranks = compute_rank_array(F, fronts)
            cd_map = {}
            for fi, front in enumerate(fronts):
                cd_vals = crowding_distance(F, front)
                cd_map[fi] = {front[i]: cd_vals[i] for i in range(len(front))}

            # Compute HV contributions for guidance
            nd_indices = fronts[0] if fronts else []
            hv_contribs = hv_marginal_contribution(F, nd_indices)
            
            # Generate offspring
            n_offspring = min(pop_size, budget - used)
            offspring_X = np.zeros((n_offspring, dim))
            offspring_F = np.zeros((n_offspring, n_obj))
            n_off = 0

            for _ in range(n_offspring):
                if used >= budget or time.time() > deadline:
                    break

                # Select parents via tournament
                p1, p2 = select_parents(F, X, ranks, cd_map)

                # Crossover (BLX-alpha style)
                if rng.random() < 0.9:
                    alpha = 0.5
                    for d in range(dim):
                        if p1 != p2:
                            lo_c = min(X[p1, d], X[p2, d]) - alpha * abs(X[p1, d] - X[p2, d])
                            hi_c = max(X[p1, d], X[p2, d]) + alpha * abs(X[p1, d] - X[p2, d])
                            offspring_X[n_off, d] = rng.uniform(max(lo[d], lo_c), min(hi[d], hi_c))
                        else:
                            offspring_X[n_off, d] = X[p1, d]
                else:
                    offspring_X[n_off, :] = X[p1, :]

                # HV-guided mutation: bias toward regions with high HV contribution
                progress = used / budget
                sigma = 0.15 * (1.0 - 0.5 * progress)
                mask = rng.random(dim) < 0.7
                if np.any(mask):
                    # Add bias toward center of objective space for underexplored regions
                    center_bias = 0.02 * rng.standard_normal(np.sum(mask))
                    offspring_X[n_off, mask] += sigma * (hi - lo)[mask] * rng.standard_normal(np.sum(mask)) + center_bias
                offspring_X[n_off, :] = np.clip(offspring_X[n_off, :], lo, hi)

                offspring_F[n_off] = problem(offspring_X[n_off, :])
                used += 1
                n_off += 1

            if n_off == 0:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X[:pop_size], offspring_X[:n_off]])
            combined_F = np.vstack([F[:pop_size], offspring_F[:n_off]])

            # Non-dominated sort the combined set
            combined_fronts = fast_non_dominated_sort(combined_F)
            combined_ranks = compute_rank_array(combined_F, combined_fronts)
            
            # Select next population using rank + crowding
            combined_cd_map = {}
            for fi, front in enumerate(combined_fronts):
                cd_vals = crowding_distance(combined_F, front)
                combined_cd_map[fi] = {front[i]: cd_vals[i] for i in range(len(front))}

            next_X = np.zeros((pop_size, dim))
            next_F = np.zeros((pop_size, n_obj))
            next_count = 0
            for fi, front in enumerate(combined_fronts):
                if next_count + len(front) <= pop_size:
                    for idx in front:
                        next_X[next_count] = combined_X[idx]
                        next_F[next_count] = combined_F[idx]
                        next_count += 1
                else:
                    remaining = pop_size - next_count
                    cd_vals = crowding_distance(combined_F, front)
                    sorted_by_cd = np.argsort(-cd_vals)
                    for i in range(min(remaining, len(sorted_by_cd))):
                        idx = front[sorted_by_cd[i]]
                        next_X[next_count] = combined_X[idx]
                        next_F[next_count] = combined_F[idx]
                        next_count += 1
                    break

            X = next_X
            F = next_F

        # Phase 3: Local polish of non-dominated points with proper eval counting
        if used < budget and time.time() < deadline:
            final_fronts = fast_non_dominated_sort(F)
            nd_indices = final_fronts[0] if final_fronts else []
            
            # Sort by HV contribution (highest first) to polish most impactful points first
            hv_contribs = hv_marginal_contribution(F, nd_indices)
            if len(hv_contribs) > 0:
                sorted_nd = np.argsort(-hv_contribs)
            else:
                sorted_nd = []
            
            for idx_pos in sorted_nd:
                if used >= budget or time.time() > deadline:
                    break
                idx = nd_indices[idx_pos]
                polished_x, polished_f, evals_used = local_polish(X[idx], F[idx])
                used += evals_used
                # Check if polished point is better and replace
                if np.all(polished_f <= F[idx]) and np.any(polished_f < F[idx]):
                    X[idx] = polished_x
                    F[idx] = polished_f

        return X
# EVOLVE-BLOCK-END
