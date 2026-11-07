"""Multi-objective optimizer for RE21 with budget 200, dim 4, n_obj 2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using MO-CMA-ES with adaptive covariance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_front(points):
            n = len(points)
            if n == 0:
                return np.array([], dtype=int)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if dominates(points[j], points[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume with reference point (minimization)."""
            if len(points) == 0:
                return 0.0
            # Only count points strictly better than reference in every objective
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # Compute hypervolume
            hv = 0.0
            prev_y = ref[1]
            for i in range(len(pts)):
                x = pts[i, 0]
                y = pts[i, 1]
                if y < prev_y:
                    hv += (ref[0] - x) * (prev_y - y)
                    prev_y = y
            return hv

        # Phase 1: Structured initialization
        n_init = 60
        rng_init = np.random.default_rng(42)

        # Corner points (2^dim = 16)
        corners = np.array([[lo[d] if b == 0 else hi[d] for b in range(dim)]
                           for i in range(2 ** dim)
                           for b in [list(np.binary_repr(i, dim))]])
        # Fix: generate corners properly
        corners = np.zeros((2 ** dim, dim))
        for i in range(2 ** dim):
            for d in range(dim):
                corners[i, d] = lo[d] if (i >> (dim - 1 - d)) & 1 == 0 else hi[d]

        # Sobol-like quasi-random points
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=7)
        n_sobol = n_init - 2 ** dim
        sobol_unit = sampler.random(n_sobol)
        sobol_pts = lo + sobol_unit * (hi - lo)

        # Random points
        n_rand = max(0, n_init - len(corners) - n_sobol)
        rand_pts = lo + (hi - lo) * rng_init.random((n_rand, dim))

        X_init = np.vstack([corners, sobol_pts, rand_pts])[:n_init]
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Phase 2: MO-CMA-ES with two islands
        pop_size = 20
        lam = pop_size  # offspring per generation
        c1 = 1 / (4 * dim + 4)  # covariance adaptation rate
        cmu = 1 / (4 * dim + 4)  # rank-mu update rate
        sigma_init = 0.3 * np.mean(hi - lo)  # initial step size

        # Two islands with different initial centers
        # Island 1: center at mean of best points
        front_idx = non_dominated_front(F_init)
        if len(front_idx) > 0:
            center1 = X_init[front_idx].mean(axis=0)
        else:
            center1 = (lo + hi) / 2

        # Island 2: center at a different region
        rng_island = np.random.default_rng(123)
        center2 = lo + (hi - lo) * rng_island.random(dim)

        # All evaluated points (for final non-dominated front)
        all_X = [X_init.copy()]
        all_F = [F_init.copy()]

        def cmaes_step(center, sigma, C, F_pool, X_pool, rng):
            """One CMA-ES generation: sample, evaluate, update."""
            nonlocal used
            # Eigendecomposition for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)

            # Sample offspring
            X_off = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                x = center + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                x = np.clip(x, lo, hi)
                X_off[i] = x

            # Evaluate offspring
            F_off = np.array([problem(x) for x in X_off])
            used += lam

            # Combine with pool for selection
            X_comb = np.vstack([X_pool, X_off])
            F_comb = np.vstack([F_pool, F_off])

            # Select top lam by hypervolume contribution (greedy)
            # Sort by non-dominated rank, then by crowding distance
            nd_idx = non_dominated_front(F_comb)
            n_comb = len(F_comb)

            # Assign ranks
            ranks = np.zeros(n_comb, dtype=int)
            remaining = np.arange(n_comb)
            rank_val = 0
            while len(remaining) > 0:
                front = non_dominated_front(F_comb[remaining])
                for idx in front:
                    ranks[remaining[idx]] = rank_val
                remaining = np.delete(remaining, front)
                rank_val += 1

            # Compute crowding distance for selection
            cd = np.zeros(n_comb)
            for k in range(n_obj):
                sorted_idx = np.argsort(F_comb[:, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F_comb[:, k].max() - F_comb[:, k].min()
                if f_range > 0:
                    for i in range(1, n_comb - 1):
                        cd[sorted_idx[i]] += (F_comb[:, k][sorted_idx[i + 1]] -
                                              F_comb[:, k][sorted_idx[i - 1]]) / f_range

            # Selection: prefer low rank, high crowding distance
            # Create selection scores
            scores = ranks * 1000 - cd
            sel_idx = np.argsort(scores)[:lam]

            # Update center: weighted mean of selected
            w = np.arange(1, lam + 1, dtype=float)
            w = w / w.sum()
            new_center = (w[:, None] * X_comb[sel_idx]).sum(axis=0)
            new_center = np.clip(new_center, lo, hi)

            # Update covariance
            # Rank-1 update
            y = (X_comb[sel_idx] - center) / sigma
            p_sigma = y.mean(axis=0)
            C = (1 + c1 * (2 - np.dot(p_sigma, p_sigma) / dim)) * C + \
                c1 * (np.outer(p_sigma, p_sigma) - C)

            # Ensure positive semi-definite
            eigvals_c, eigvecs_c = np.linalg.eigh(C)
            eigvals_c = np.maximum(eigvals_c, 1e-10)
            C = (eigvecs_c * eigvals_c) @ eigvecs_c.T

            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma = sigma * np.exp(0.5 / (4*dim) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 0.5 * np.mean(hi - lo))

            return new_center, sigma, C, X_comb[sel_idx], F_comb[sel_idx]

        # Run CMA-ES on two islands
        centers = [center1, center2]
        sigmas = [sigma_init, sigma_init * 0.8]
        covs = [np.eye(dim), np.eye(dim)]
        pools = [X_init, X_init]
        F_pools = [F_init, F_init]

        rngs = [np.random.default_rng(1000 + i) for i in range(2)]

        n_gens = (budget - used) // (lam * 2)

        for gen in range(n_gens):
            if used >= budget or time.time() > deadline - 5:
                break

            for island in range(2):
                if used >= budget or time.time() > deadline - 5:
                    break

                # Occasionally reseed with best from other island
                if gen > 0 and gen % 10 == 0 and island == 1:
                    front0 = non_dominated_front(F_pools[0])
                    if len(front0) > 0:
                        idx = front0[np.argmin(np.sum(F_pools[0][front0], axis=1))]
                        centers[island] = X_pools[0][idx] * 0.5 + centers[island] * 0.5
                        sigmas[island] = sigma_init * 0.5

                centers[island], sigmas[island], covs[island], \
                    pools[island], F_pools[island] = cmaes_step(
                        centers[island], sigmas[island], covs[island],
                        pools[island], F_pools[island], rngs[island])

                # Record all evaluated points
                all_X.append(pools[island])
                all_F.append(F_pools[island])

        # Combine all evaluated points and find non-dominated front
        all_X = np.vstack(all_X)
        all_F = np.vstack(all_F)

        # Remove duplicates (approximate)
        unique_mask = np.ones(len(all_F), dtype=bool)
        for i in range(len(all_F)):
            if not unique_mask[i]:
                continue
            for j in range(i + 1, len(all_F)):
                if not unique_mask[j]:
                    continue
                if np.allclose(all_F[i], all_F[j], atol=1e-10):
                    unique_mask[j] = False

        all_X = all_X[unique_mask]
        all_F = all_F[unique_mask]

        final_front = non_dominated_front(all_F)

        return all_X[final_front]
# EVOLVE-BLOCK-END
