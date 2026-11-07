"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE37: LHS init + NSGA-II + hypervolume-gap adaptive search + scalarized polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (fixed per task)
        ref = np.array([1.1, 1.3, 1.1])

        # Use the global numpy RNG (seeded by evaluator via np.random.seed)
        class _RNG:
            def __init__(self):
                pass
            def permutation(self, n):
                return np.random.permutation(n)
            def random(self, n=None):
                if n is None:
                    return np.random.random()
                return np.random.random(n)
            def integers(self, n):
                return np.random.randint(0, n)
            def standard_normal(self, size=None):
                return np.random.standard_normal(size)
        rng = _RNG()

        # Track all evaluated points
        X_all = []
        F_all = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count += 1
            return f

        def time_ok():
            return (time.time() - start_time) < deadline and eval_count < budget

        # 1. Latin Hypercube Sampling for initial population
        def lhs_sample(n, d, lo, hi, rng):
            samples = np.zeros((n, d))
            for dd in range(d):
                perm = rng.permutation(n)
                samples[:, dd] = (perm + rng.random(n)) / n
            return lo + (hi - lo) * samples

        pop_size = 40
        X = lhs_sample(pop_size, dim, lo, hi, rng)
        for x in X:
            if not time_ok():
                break
            evaluate(x)

        # 2. Fast non-dominated sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            ranks = np.zeros(n, dtype=int)
            front = []

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[p] += 1
                if n_dom[p] == 0:
                    ranks[p] = 0
                    front.append(p)

            i = 0
            while len(front) > 0:
                i += 1
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            ranks[q] = i
                            next_front.append(q)
                front = next_front

            return ranks

        # 3. Crowding distance
        def crowding_distance(F, ranks):
            n = len(F)
            cd = np.zeros(n)
            unique_ranks = np.unique(ranks)

            for rank in unique_ranks:
                members = np.where(ranks == rank)[0]
                if len(members) <= 2:
                    cd[members] = np.inf
                    continue

                for m in range(n_obj):
                    obj_vals = F[members, m]
                    sorted_idx = np.argsort(obj_vals)
                    cd[members[sorted_idx[0]]] = np.inf
                    cd[members[sorted_idx[-1]]] = np.inf
                    obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                    if obj_range > 1e-12:
                        for k in range(1, len(sorted_idx) - 1):
                            cd[members[sorted_idx[k]]] += (
                                obj_vals[sorted_idx[k + 1]] - obj_vals[sorted_idx[k - 1]]
                            ) / obj_range

            return cd

        # 4. SBX crossover
        def sbx_crossover(x1, x2, beta=15):
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for i in range(dim):
                if x1[i] == x2[i]:
                    child1[i] = x1[i]
                    child2[i] = x2[i]
                    continue
                u = rng.random()
                if u <= 0.5:
                    beta_c = (2 * u) ** (-1) if u > 0 else 1e6
                    beta_c = max(beta_c, 1e-6)
                    child1[i] = 0.5 * ((1 + beta_c) * x1[i] + (1 - beta_c) * x2[i])
                else:
                    beta_c = (2 * (1 - u)) ** (-1) if u < 1 else 1e6
                    beta_c = max(beta_c, 1e-6)
                    child1[i] = 0.5 * ((1 + beta_c) * x2[i] + (1 - beta_c) * x1[i])
                child2[i] = hi[i] + lo[i] - child1[i]
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        # 5. Mutation
        def mutate(child, sigma_scale=0.1):
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    sigma = sigma_scale * (hi[i] - lo[i])
                    child[i] += sigma * rng.standard_normal()
                    child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        # 6. Hypervolume contribution calculation for gap-filling
        def hypervolume_contribution(F, ref):
            """Compute the hypervolume contribution of each point in the non-dominated set.
            Uses a simple incremental approach for 3 objectives."""
            n = len(F)
            if n == 0:
                return np.zeros(0)
            
            # Only consider points that are better than reference in all objectives
            valid_mask = np.all(F < ref, axis=1)
            if not np.any(valid_mask):
                return np.zeros(n)
            
            # For each point, compute its hypervolume contribution
            # This is the volume of the region dominated by this point but not by any other point
            contributions = np.zeros(n)
            
            for i in range(n):
                if not valid_mask[i]:
                    continue
                
                # The region dominated by point i is [F[i], ref]
                # Subtract regions dominated by other points that are better than i
                # A point j dominates i if F[j] <= F[i] and F[j] != F[i]
                
                # Compute the volume of [F[i], ref]
                vol_i = np.prod(ref - F[i])
                
                # Subtract overlapping volumes with points that dominate i
                # For simplicity, we use an approximation: for each point j that dominates i,
                # subtract the volume of the intersection [F[j], ref] ∩ [F[i], ref] = [F[j], ref] (since F[j] <= F[i])
                # But this over-subtracts. A better approach is to use the standard HV computation.
                
                # For now, use a simple heuristic: the contribution is proportional to how "extreme" the point is
                # This is not exact but serves as a guide for gap-filling
                contributions[i] = vol_i
            
            # Normalize to [0, 1]
            if contributions.max() > 0:
                contributions = contributions / contributions.max()
            
            return contributions

        # 7. Main NSGA-II evolution loop
        G = (budget - eval_count) // pop_size
        for gen in range(G):
            if not time_ok():
                break

            # Get current population from recent evaluations
            F = np.array(F_all[-pop_size:])
            X = np.array(X_all[-pop_size:])

            ranks = fast_non_dominated_sort(F)
            cd = crowding_distance(F, ranks)

            offspring_X = []
            offspring_F = []

            for _ in range(pop_size // 2):
                if not time_ok():
                    break

                # Tournament selection
                def tournament_select(ranks, cd, k=3):
                    best_idx = rng.integers(pop_size)
                    for _ in range(k - 1):
                        cand = rng.integers(pop_size)
                        if ranks[cand] < ranks[best_idx]:
                            best_idx = cand
                        elif ranks[cand] == ranks[best_idx] and cd[cand] > cd[best_idx]:
                            best_idx = cand
                    return best_idx

                p1 = tournament_select(ranks, cd)
                p2 = tournament_select(ranks, cd)

                if rng.random() < 0.9:
                    c1, c2 = sbx_crossover(X[p1], X[p2])
                else:
                    c1, c2 = X[p1].copy(), X[p2].copy()

                c1 = mutate(c1)
                c2 = mutate(c2)

                f1 = evaluate(c1)
                f2 = evaluate(c2)

                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(f2)

            if not time_ok():
                break

            # Combine and select
            X_new = np.vstack([X, np.array(offspring_X)])
            F_new = np.vstack([F, np.array(offspring_F)])

            ranks_new = fast_non_dominated_sort(F_new)
            cd_new = crowding_distance(F_new, ranks_new)

            sort_idx = np.lexsort((-cd_new, ranks_new))
            # Update X_all and F_all with the selected population
            # We keep the best pop_size points
            selected_X = X_new[sort_idx[:pop_size]]
            selected_F = F_new[sort_idx[:pop_size]]
            
            # Replace the last pop_size entries in X_all and F_all
            X_all[-pop_size:] = selected_X
            F_all[-pop_size:] = selected_F

        # 8. Hypervolume-driven adaptive gap-filling phase
        if time_ok():
            # Get the current non-dominated set
            F_all_arr = np.array(F_all)
            X_all_arr = np.array(X_all)
            
            ranks_all = fast_non_dominated_sort(F_all_arr)
            nondom_mask = (ranks_all == 0)
            nondom_idx = np.where(nondom_mask)[0]
            
            if len(nondom_idx) > 0:
                F_nd = F_all_arr[nondom_idx]
                X_nd = X_all_arr[nondom_idx]
                
                # Compute hypervolume contributions to identify gaps
                # Use a simple approach: for each non-dominated point, estimate its "gap" 
                # by looking at the distance to its neighbors in objective space
                n_nd = len(F_nd)
                gap_scores = np.zeros(n_nd)
                
                for i in range(n_nd):
                    # Compute the minimum distance to other non-dominated points in objective space
                    min_dist = np.inf
                    for j in range(n_nd):
                        if i == j:
                            continue
                        dist = np.linalg.norm(F_nd[i] - F_nd[j])
                        min_dist = min(min_dist, dist)
                    # Larger minimum distance means larger gap
                    gap_scores[i] = min_dist
                
                # Sort by gap score (largest gap first)
                gap_order = np.argsort(gap_scores)[::-1]
                
                # For the top gaps, try to fill them by sampling in the objective space
                # and mapping back to the decision space using a simple approach
                max_gap_fills = min(30, budget - eval_count)
                for k in range(max_gap_fills):
                    if not time_ok():
                        break
                    
                    # Pick the point with the largest gap
                    idx = gap_order[k % n_nd]
                    F_target = F_nd[idx]
                    
                    # Generate candidate points near the target in objective space
                    # We'll use a random perturbation in the decision space and check if it improves the gap
                    for attempt in range(3):
                        if not time_ok():
                            break
                        
                        # Random perturbation around the point with the largest gap
                        x_base = X_nd[idx]
                        scale = 0.1 * (hi - lo)
                        x_candidate = x_base + scale * rng.standard_normal(dim)
                        x_candidate = np.clip(x_candidate, lo, hi)
                        
                        f_candidate = evaluate(x_candidate)
                        
                        # Check if this point is non-dominated
                        is_dom = False
                        for j in range(len(F_all)):
                            if np.all(F_all[j] <= f_candidate) and np.any(F_all[j] < f_candidate):
                                is_dom = True
                                break
                        
                        if not is_dom:
                            # This point improves the non-dominated set
                            # Update gap scores
                            gap_scores[idx] = 0  # Mark as filled
                            break

        # 9. Weight-vector based local refinement around non-dominated points
        if time_ok():
            F_all_arr = np.array(F_all)
            X_all_arr = np.array(X_all)
            ranks_final = fast_non_dominated_sort(F_all_arr)
            nondom_mask = (ranks_final == 0)
            nondom_idx = np.where(nondom_mask)[0]

            # Generate weight vectors for scalarization
            def generate_weights(n_obj, n_levels=3):
                weights = []
                if n_obj == 3:
                    for i in range(n_levels + 1):
                        for j in range(n_levels + 1 - i):
                            k = n_levels - i - j
                            w = np.array([i, j, k], dtype=float)
                            if np.sum(w) > 0:
                                weights.append(w / np.sum(w))
                    return np.array(weights)
                else:
                    return np.array([np.ones(n_obj) / n_obj])

            weight_vectors = generate_weights(n_obj, n_levels=4)

            # For each weight vector, do a local search from the best non-dominated point
            # that aligns with that weight direction
            for w in weight_vectors:
                if not time_ok():
                    break

                # Find the non-dominated point that best matches this weight direction
                if len(nondom_idx) > 0:
                    scalarized = F_all_arr[nondom_idx] @ w
                    best_local_idx = nondom_idx[np.argmin(scalarized)]
                else:
                    continue

                base = X_all_arr[best_local_idx].copy()

                # Local search: try small perturbations
                for step in range(4):
                    if not time_ok():
                        break

                    scale = 0.05 * (0.7 ** step) * (hi - lo)
                    perturbed = base + scale * rng.standard_normal(dim)
                    perturbed = np.clip(perturbed, lo, hi)
                    fp = evaluate(perturbed)

                    scalar_new = fp @ w
                    scalar_base = F_all_arr[best_local_idx] @ w

                    if scalar_new < scalar_base:
                        is_dom = False
                        for j in range(len(F_all)):
                            if np.all(F_all[j] <= fp) and np.any(F_all[j] < fp):
                                is_dom = True
                                break
                        if not is_dom:
                            F_all[best_local_idx] = fp
                            X_all[best_local_idx] = perturbed
                            base = perturbed.copy()
                    else:
                        perturbed2 = base + scale * rng.standard_normal(dim)
                        perturbed2 = np.clip(perturbed2, lo, hi)
                        fp2 = evaluate(perturbed2)
                        scalar_new2 = fp2 @ w
                        if scalar_new2 < scalar_base:
                            is_dom2 = False
                            for j in range(len(F_all)):
                                if np.all(F_all[j] <= fp2) and np.any(F_all[j] < fp2):
                                    is_dom2 = True
                                    break
                            if not is_dom2:
                                F_all[best_local_idx] = fp2
                                X_all[best_local_idx] = perturbed2
                                base = perturbed2.copy()

        # 10. High-precision scalarized polish pass (Nelder-Mead on smooth scalarizations)
        if time_ok():
            # Re-compute non-dominated set after local search
            F_all_arr = np.array(F_all)
            X_all_arr = np.array(X_all)
            ranks_final = fast_non_dominated_sort(F_all_arr)
            nondom_mask = (ranks_final == 0)
            nondom_idx = np.where(nondom_mask)[0]

            # Build a richer set of weight vectors (simplex grid, 5 levels) for polish
            def generate_weights_rich(n_obj, n_levels=5):
                weights = []
                if n_obj == 3:
                    for i in range(n_levels + 1):
                        for j in range(n_levels + 1 - i):
                            k = n_levels - i - j
                            w = np.array([i, j, k], dtype=float)
                            if np.sum(w) > 0:
                                weights.append(w / np.sum(w))
                    return np.array(weights)
                else:
                    return np.array([np.ones(n_obj) / n_obj])

            weight_vectors = generate_weights_rich(n_obj, n_levels=5)

            # Limit polish work to a subset of weight vectors to stay within budget/time
            max_polish_vectors = 12
            if len(weight_vectors) > max_polish_vectors:
                # Uniformly sample weight vectors
                idx = np.linspace(0, len(weight_vectors) - 1, max_polish_vectors, dtype=int)
                weight_vectors = weight_vectors[idx]

            # For each weight vector, run Nelder-Mead from the best-aligned non-dominated point
            for w in weight_vectors:
                if not time_ok():
                    break

                if len(nondom_idx) == 0:
                    break

                # Best-aligned non-dominated point for this weight
                scalarized = F_all_arr[nondom_idx] @ w
                best_idx = nondom_idx[np.argmin(scalarized)]
                x0 = X_all_arr[best_idx].copy()

                # Budget cap per polish run: at most 60 function evaluations
                per_run_budget = min(60, budget - eval_count)
                if per_run_budget < 5:
                    break

                # Smooth Tchebycheff scalarization (log-sum-exp soft-max)
                tau = 20.0

                def scalar_objective(x, w=w):
                    fx = problem(x)
                    fx = np.asarray(fx, dtype=float)
                    weighted = w * fx
                    # Numerically stable log-sum-exp
                    m = np.max(weighted)
                    if m > 700:  # avoid overflow
                        return m + (1.0 / tau) * np.log(np.sum(np.exp(weighted - m)))
                    return (1.0 / tau) * np.log(np.sum(np.exp(weighted)))

                # Nelder-Mead with tight tolerances for a high-precision polish
                try:
                    res = minimize(
                        scalar_objective,
                        x0,
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={
                            'maxiter': per_run_budget,
                            'maxfev': per_run_budget,
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                            'adaptive': True,
                        }
                    )
                    # Count the evaluations actually used by Nelder-Mead
                    nfev = getattr(res, 'nfev', 0)
                    eval_count += int(nfev)

                    # Evaluate the polished point
                    x_polished = np.clip(res.x, lo, hi)
                    f_polished = problem(x_polished)
                    eval_count += 1

                    # Accept if it is not dominated by any existing point
                    is_dom = False
                    for j in range(len(F_all)):
                        if np.all(F_all[j] <= f_polished) and np.any(F_all[j] < f_polished):
                            is_dom = True
                            break
                    if not is_dom:
                        # Replace the best-aligned point (or add if we have room)
                        F_all[best_idx] = f_polished
                        X_all[best_idx] = x_polished
                except Exception:
                    # If polish fails, keep the existing point
                    pass

        return np.array(X_all)
# EVOLVE-BLOCK-END
