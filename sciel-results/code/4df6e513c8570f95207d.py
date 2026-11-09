"""Multi-objective optimizer for RE34 using Hypervolume-Estimated Descent (HVED)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HVED-based multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Reference point for hypervolume (from task contract)
        ref = np.array([1705.0, 12.0, 0.3])

        # Population size: use a moderate size to allow multiple generations
        pop_size = min(50, self.budget // 4)
        if pop_size < 15:
            pop_size = 15

        # Initial population: stratified Latin hypercube + corners
        X = np.empty((pop_size, dim))

        # Add some structured points for initial coverage
        n_structured = min(2 * dim + 1, pop_size)
        structured = []
        structured.append(lo.copy())
        structured.append(hi.copy())
        mid = (lo + hi) / 2.0
        structured.append(mid.copy())
        for i in range(dim):
            pt = mid.copy()
            pt[i] = lo[i]
            structured.append(pt.copy())
            pt = mid.copy()
            pt[i] = hi[i]
            structured.append(pt.copy())

        structured = np.array(structured[:n_structured])
        X[:n_structured] = structured

        # Latin hypercube for the rest
        if pop_size > n_structured:
            n_lhc = pop_size - n_structured
            # Latin hypercube sampling
            lhc = np.zeros((n_lhc, dim))
            for d in range(dim):
                perm = np.random.permutation(n_lhc)
                for i in range(n_lhc):
                    lhc[i, d] = (perm[i] + np.random.random()) / n_lhc
            X[n_structured:] = lo + (hi - lo) * lhc

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        evaluated = pop_size

        def hv_contribution_3d(F_arr, ref_pt):
            """
            Compute approximate hypervolume contribution of each point in a 3D objective space.
            Uses the reference point to determine which region each point 'owns'.
            A point's contribution is the volume of the box it uniquely covers.
            """
            n = len(F_arr)
            if n == 0:
                return np.array([])

            # Filter: only points strictly better than reference in all objectives
            valid_mask = np.all(F_arr < ref_pt, axis=1)
            contributions = np.zeros(n)

            if not np.any(valid_mask):
                return contributions

            valid_idx = np.where(valid_mask)[0]
            F_valid = F_arr[valid_idx]
            n_valid = len(F_valid)

            # For 3D HV contribution, we can use a simplified approach:
            # Sort by first objective, then compute incremental volumes
            # This is an approximation but captures the key idea

            # Normalize to [0,1] for stable computation
            f_min = np.min(F_valid, axis=0)
            f_max = np.max(F_valid, axis=0)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0

            # Use a grid-based approximation for HV contribution
            # For each point, estimate the volume it contributes
            # by checking how many other points dominate it in subsets of objectives

            # Simpler approach: rank-based contribution
            # Points that are non-dominated and spread out get higher contribution
            # Compute a simple dominance-based score

            # For each valid point, compute how many other valid points it dominates
            # and how many dominate it
            dom_count = np.zeros(n_valid, dtype=int)
            dominated_by = np.zeros(n_valid, dtype=int)

            for i in range(n_valid):
                for j in range(n_valid):
                    if i == j:
                        continue
                    if np.all(F_valid[i] <= F_valid[j]) and np.any(F_valid[i] < F_valid[j]):
                        dom_count[i] += 1
                    elif np.all(F_valid[j] <= F_valid[i]) and np.any(F_valid[j] < F_valid[i]):
                        dominated_by[i] += 1

            # Contribution: high if it dominates many and is dominated by few
            # Scale by the actual objective ranges for meaningful HV units
            base_vol = np.prod(ref_pt - F_valid, axis=1)
            # Weight by non-dominance quality
            quality = (dom_count + 1) / (dominated_by + 1)
            contributions_valid = base_vol * quality

            # Normalize to [0, 1] for selection
            max_contrib = np.max(contributions_valid)
            if max_contrib > 0:
                contributions_valid = contributions_valid / max_contrib

            for i, idx in enumerate(valid_idx):
                contributions[idx] = contributions_valid[i]

            return contributions

        def de_vectorized(X_pop, F_pop, ref_pt):
            """
            Vectorized Differential Evolution with hypervolume-based parent selection.
            Returns offspring population.
            """
            n = len(X_pop)
            # Compute HV contributions for parent selection
            hv_contrib = hv_contribution_3d(F_pop, ref_pt)

            # Select parents based on HV contribution (higher is better)
            # Use weighted random selection
            weights = hv_contrib + 0.01  # small epsilon to avoid zero weights
            weights = weights / np.sum(weights)

            # DE/rand/1: offspring = x_a + F*(x_b - x_c)
            a_idx = np.random.choice(n, size=n, replace=True, p=weights)
            b_idx = np.random.choice(n, size=n, replace=True, p=weights)
            c_idx = np.random.choice(n, size=n, replace=True, p=weights)

            # Avoid selecting the same individual for all three
            # (not critical but helps diversity)
            for i in range(n):
                while b_idx[i] == a_idx[i] or c_idx[i] == a_idx[i] or c_idx[i] == b_idx[i]:
                    b_idx[i] = np.random.choice(n)
                    c_idx[i] = np.random.choice(n)

            # Differential weight (adaptive: start higher, decrease)
            F_de = 0.8

            # Vectorized DE
            offspring = X_pop[a_idx] + F_de * (X_pop[b_idx] - X_pop[c_idx])

            # Add Gaussian mutation for exploration
            mutation_scale = 0.1 * (hi - lo)
            offspring += np.random.normal(0, 1, size=offspring.shape) * mutation_scale

            # Clip to bounds
            offspring = np.clip(offspring, lo, hi)

            return offspring

        # Main HVED loop
        while evaluated + pop_size <= self.budget:
            if time.time() > deadline:
                break

            # Generate offspring using DE with HV-based selection
            offspring_X = de_vectorized(X, F, ref)

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            evaluated += pop_size

            # Combine and select next generation
            # Keep the best pop_size points from combined population based on HV
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Compute HV contributions for all combined points
            hv_all = hv_contribution_3d(combined_F, ref)

            # Select top pop_size by HV contribution
            # But also ensure diversity: use a greedy selection that maximizes HV
            # For simplicity, select by HV contribution with some randomness
            n_select = pop_size
            if len(hv_all) <= n_select:
                X = combined_X.copy()
                F = combined_F.copy()
            else:
                # Weighted selection with diversity consideration
                # First, identify non-dominated points (they should be prioritized)
                # Simple non-domination check
                n_combined = len(combined_F)
                is_nondom = np.ones(n_combined, dtype=bool)
                for i in range(n_combined):
                    if not is_nondom[i]:
                        continue
                    for j in range(n_combined):
                        if i == j:
                            continue
                        if np.all(combined_F[j] <= combined_F[i]) and np.any(combined_F[j] < combined_F[i]):
                            is_nondom[i] = False
                            break

                # Prioritize non-dominated points, then by HV contribution
                scores = hv_all.copy()
                scores[~is_nondom] *= 0.5  # penalize dominated points

                # Select top n_select with some randomness
                # Use top-k with temperature
                top_k = min(n_select * 2, len(scores))
                top_indices = np.argsort(scores)[-top_k:]
                # Add some randomness to the selection
                if len(top_indices) > n_select:
                    # Weighted random from top candidates
                    top_scores = scores[top_indices]
                    top_weights = top_scores + 0.01
                    top_weights = top_weights / np.sum(top_weights)
                    selected_local = np.random.choice(len(top_indices), size=n_select, replace=False, p=top_weights)
                    selected_indices = top_indices[selected_local]
                else:
                    selected_indices = top_indices

                X = combined_X[selected_indices]
                F = combined_F[selected_indices]

        return X
# EVOLVE-BLOCK-END
