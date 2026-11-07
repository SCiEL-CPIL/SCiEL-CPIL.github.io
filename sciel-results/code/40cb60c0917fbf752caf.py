# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34: HV-guided local search with LHS init and efficient polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_pareto(self, F):
        """Vectorized Pareto filter: returns boolean mask of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        if n == 1:
            return np.array([True])
        # F[i] is dominated if exists j: F[j] <= F[i] and F[j] < F[i]
        # F[j] <= F[i] for all objectives: (F[j] - F[i]) <= 0 for all obj
        # F[j] < F[i] for at least one objective: min(F[j] - F[i]) < 0
        diff = F[:, np.newaxis, :] - F[np.newaxis, :, :]  # diff[i,j] = F[j] - F[i]
        # dominated[i] = any over j != i of (all(diff[i,j] <= 0) and any(diff[i,j] < 0))
        all_leq = np.all(diff <= 0, axis=2)  # all_leq[i,j] = True if F[j] <= F[i]
        any_lt = np.any(diff < 0, axis=2)    # any_lt[i,j] = True if F[j] < F[i] in at least one obj
        dominated = (all_leq & any_lt)
        np.fill_diagonal(dominated, False)
        return ~dominated.any(axis=1)

    def _hv_contribution(self, F, ref):
        """Compute HV contribution of each point in F (3D case)."""
        n = len(F)
        if n == 0:
            return np.array([])
        # For 3D, HV contribution can be computed via inclusion-exclusion
        # Simpler: compute total HV with and without each point (expensive but accurate)
        # For speed, use a proxy: volume of box from point to ref, adjusted by neighbors
        # Fast proxy: (ref - F).prod(axis=1) as upper bound on contribution
        return np.prod(np.maximum(ref - F, 0), axis=1)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        rng = np.random

        # --- Phase 1: Initial sampling with LHS + random ---
        n_lhs = 25
        n_rand = 15
        n_init = min(n_lhs + n_rand, budget)

        X_list = []
        lhs_X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            lhs_X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + np.random.uniform(size=n_lhs)) / n_lhs
        lhs_X = np.clip(lhs_X, lo, hi)
        X_list.append(lhs_X)

        rand_X = lo + np.random.uniform(size=(n_rand, dim)) * (hi - lo)
        rand_X = np.clip(rand_X, lo, hi)
        X_list.append(rand_X)

        X_init = np.vstack(X_list)
        F_init = np.array([problem(x) for x in X_init])
        used = len(X_init)

        all_X = [X_init]
        all_F = [F_init]

        # Build initial Pareto archive
        valid = np.all(F_init < ref, axis=1)
        if np.any(valid):
            pF = F_init[valid]
            pX = X_init[valid]
            p_mask = self._is_pareto(pF)
            pareto_X = pX[p_mask].copy()
            pareto_F = pF[p_mask].copy()
        else:
            pareto_X = np.empty((0, dim))
            pareto_F = np.empty((0, n_obj))

        # --- Phase 2: Iterative improvement with adaptive mutation ---
        max_archive = 60

        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget

            # Parent selection: weighted by HV contribution proxy
            if len(pareto_F) > 0:
                # Use HV contribution proxy for weighting
                hv_contribs = self._hv_contribution(pareto_F, ref)
                # Add small epsilon to avoid zero weights
                weights = hv_contribs + 1e-6
                # Temperature: higher early for exploration, lower late for exploitation
                temp = 2.0 * (1.0 - progress) + 0.5
                weights = np.power(weights, 1.0 / temp)
                probs = weights / weights.sum()
                parent_idx = np.random.choice(len(pareto_F), p=probs)
                parent = pareto_X[parent_idx]
            else:
                parent = lo + np.random.uniform(size=dim) * (hi - lo)

            # Adaptive mutation with smoother schedule
            if progress < 0.25:
                base_step = 0.7
                use_coord = np.random.uniform() < 0.15
            elif progress < 0.5:
                base_step = 0.45
                use_coord = np.random.uniform() < 0.4
            elif progress < 0.8:
                base_step = 0.25
                use_coord = np.random.uniform() < 0.65
            else:
                base_step = 0.12
                use_coord = np.random.uniform() < 0.85

            step = base_step * (0.5 + np.random.uniform())

            if not use_coord:
                child = parent + step * (hi - lo) * np.random.standard_normal(dim)
            else:
                child = parent.copy()
                n_perturb = np.random.randint(1, 3)
                coords = np.random.choice(dim, size=n_perturb, replace=False)
                for c in coords:
                    child[c] = parent[c] + step * (hi[c] - lo[c]) * np.random.standard_normal()
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            all_X.append(np.array([child]))
            all_F.append(np.array([fc]))

            # Update Pareto archive efficiently
            if np.all(fc < ref):
                # Check if dominated by any existing point
                if len(pareto_F) > 0:
                    diff = pareto_F - fc  # diff[i] = pareto_F[i] - fc
                    dominated = np.any(np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1))
                else:
                    dominated = False

                if not dominated:
                    pareto_X = np.vstack([pareto_X, [child]])
                    pareto_F = np.vstack([pareto_F, [fc]])
                    # Remove points dominated by new point
                    if len(pareto_F) > 1:
                        diff2 = fc - pareto_F[:-1]  # diff2[i] = fc - pareto_F[i]
                        keep = ~(np.all(diff2 >= 0, axis=1) & np.any(diff2 > 0, axis=1))
                        keep = np.concatenate([keep, [True]])
                        pareto_X = pareto_X[keep]
                        pareto_F = pareto_F[keep]

                    # Prune archive if too large
                    if len(pareto_F) > max_archive:
                        p_mask = self._is_pareto(pareto_F)
                        pareto_X = pareto_X[p_mask]
                        pareto_F = pareto_F[p_mask]
                        if len(pareto_F) > max_archive:
                            # Keep points with highest HV contribution
                            hv_c = self._hv_contribution(pareto_F, ref)
                            top_idx = np.argsort(-hv_c)[:max_archive]
                            pareto_X = pareto_X[top_idx]
                            pareto_F = pareto_F[top_idx]

        # --- Phase 3: High-precision coordinate descent polish ---
        if used < budget and len(pareto_F) > 0:
            # Sort by HV contribution (higher = more valuable)
            hv_c = self._hv_contribution(pareto_F, ref)
            top_k = min(8, len(pareto_F))
            top_idx = np.argsort(-hv_c)[:top_k]

            # Multi-pass polish with decreasing step sizes
            step_sizes = [0.08, 0.04, 0.02, 0.01, 0.005]

            for pi in top_idx:
                if used >= budget or time.time() > deadline - 10:
                    break
                x_cur = pareto_X[pi].copy()
                f_cur = problem(x_cur)
                used += 1

                # Multi-scale coordinate descent
                for ss in step_sizes:
                    if used >= budget or time.time() > deadline - 5:
                        break
                    improved = True
                    while improved and used < budget and time.time() < deadline - 5:
                        improved = False
                        for coord in range(dim):
                            if used >= budget or time.time() > deadline - 5:
                                break
                            for direction in [-1, 1]:
                                if used >= budget:
                                    break
                                x_try = x_cur.copy()
                                x_try[coord] += direction * ss * (hi[coord] - lo[coord])
                                x_try = np.clip(x_try, lo, hi)

                                f_try = problem(x_try)
                                used += 1

                                # Accept if it improves any objective without worsening others
                                if np.all(f_try <= f_cur) and np.any(f_try < f_cur):
                                    x_cur = x_try
                                    f_cur = f_try
                                    improved = True
                                elif np.all(f_try < ref):
                                    # Update archive with this point
                                    all_X.append(np.array([x_try]))
                                    all_F.append(np.array([f_try]))

                                    if len(pareto_F) > 0:
                                        diff = pareto_F - f_try
                                        dominated = np.any(np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1))
                                    else:
                                        dominated = False

                                    if not dominated:
                                        pareto_X = np.vstack([pareto_X, [x_try]])
                                        pareto_F = np.vstack([pareto_F, [f_try]])
                                        if len(pareto_F) > 1:
                                            diff2 = f_try - pareto_F[:-1]
                                            keep = ~(np.all(diff2 >= 0, axis=1) & np.any(diff2 > 0, axis=1))
                                            keep = np.concatenate([keep, [True]])
                                            pareto_X = pareto_X[keep]
                                            pareto_F = pareto_F[keep]

                                        if len(pareto_F) > max_archive:
                                            p_mask = self._is_pareto(pareto_F)
                                            pareto_X = pareto_X[p_mask]
                                            pareto_F = pareto_F[p_mask]
                                            if len(pareto_F) > max_archive:
                                                hv_c2 = self._hv_contribution(pareto_F, ref)
                                                top_idx2 = np.argsort(-hv_c2)[:max_archive]
                                                pareto_X = pareto_X[top_idx2]
                                                pareto_F = pareto_F[top_idx2]

        # Combine all evaluated points
        all_X = np.vstack(all_X)
        all_F = np.vstack(all_F)

        return all_X

# EVOLVE-BLOCK-END
