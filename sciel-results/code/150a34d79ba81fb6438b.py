"""Surrogate-based Bayesian optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """GP-surrogate Bayesian optimization with local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s
        rng = np.random.default_rng(42)
        dim = self.dim
        lo, hi = -5.0, 5.0

        # --- Phase 1: Initial diverse sampling ---
        # Latin Hypercube + center
        n_lhs = 30
        # LHS
        perm = [rng.permutation(n_lhs) for _ in range(dim)]
        lhs_points = np.array([(perm[d] + rng.random()) / n_lhs for d in range(dim)]).T
        lhs_points = lo + lhs_points * (hi - lo)
        # Add center and a few axis points
        extra = np.array([np.zeros(dim)])
        for d in range(dim):
            for val in [-3.0, 3.0]:
                p = np.zeros(dim)
                p[d] = val
                extra = np.vstack([extra, p])
        # Deduplicate
        init_points = np.vstack([lhs_points, extra])
        # Remove duplicates
        unique_pts, unique_idx = np.unique(np.round(init_points, 6), axis=0, return_index=True)
        init_points = init_points[unique_idx]
        # Cap initial points to leave budget
        n_init = min(len(init_points), 40)
        init_points = init_points[:n_init]

        X = []
        Y = []
        best_x = None
        best_f = np.inf

        for i in range(n_init):
            if time.time() > deadline:
                break
            x = init_points[i]
            x = np.clip(x, lo, hi)
            f = func(x)
            X.append(x)
            Y.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 2: GP surrogate + EI acquisition ---
        calls_made = n_init
        remaining = self.budget - calls_made

        # GP parameters
        length_scale = 2.0
        sigma_f = 1.0
        noise = 1e-6

        def rbf_kernel(X1, X2, ls):
            D = np.sum(X1 ** 2, axis=1)[:, None] - 2.0 * X1 @ X2.T + np.sum(X2 ** 2, axis=1)[None, :]
            D = np.maximum(D, 0)
            return np.exp(-D / (2.0 * ls * ls))

        def gp_predict(X_train, Y_train, X_test, ls, sf, noise):
            K = rbf_kernel(X_train, X_train, ls) + noise * np.eye(len(X_train))
            K = K + 1e-8 * np.eye(len(X_train))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X_train)))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            K_s = rbf_kernel(X_test, X_train, ls)
            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = sf * sf - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 1e-10)
            return mu, np.sqrt(var)

        def expected_improvement(mu, sigma, best_f, xi=0.01):
            sigma = np.maximum(sigma, 1e-8)
            z = (best_f - xi - mu) / sigma
            from scipy.stats import norm
            ei = (best_f - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Adaptive EI exploration: start with wider exploration, tighten over time
        n_surrogate = 0
        n_refine = 0
        use_local_refine = False

        while calls_made < self.budget and time.time() < deadline:
            if len(X) < 2:
                # Not enough points, sample randomly
                x_new = rng.uniform(lo, hi, dim)
            else:
                X_arr = np.array(X)
                Y_arr = np.array(Y)
                # Grid for EI maximization: coarse grid + local grid around best
                # Coarse grid
                n_grid = 12
                grid_axes = [np.linspace(lo, hi, n_grid) for _ in range(dim)]
                # Create a reduced grid to keep size manageable
                # Use a random subset of grid points + local neighborhood
                n_candidates = 500
                # Random points in box
                cand = rng.uniform(lo, hi, (n_candidates, dim))
                # Local neighborhood around best
                if best_x is not None:
                    local_scale = max(0.5, (hi - lo) / 10.0)
                    local = best_x + rng.uniform(-local_scale, local_scale, (200, dim))
                    local = np.clip(local, lo, hi)
                    cand = np.vstack([cand, local])
                    # Also add axis-aligned perturbations
                    for d in range(dim):
                        for delta in [-1.0, -0.5, 0.5, 1.0]:
                            p = best_x.copy()
                            p[d] += delta
                            p = np.clip(p, lo, hi)
                            cand = np.vstack([cand, p[None, :]])
                cand = np.clip(cand, lo, hi)
                # Deduplicate candidates
                cand_unique, _ = np.unique(np.round(cand, 5), axis=0, return_index=True)
                cand = cand_unique
                if len(cand) == 0:
                    cand = rng.uniform(lo, hi, (10, dim))

                # Evaluate EI on candidates
                mu, sigma = gp_predict(X_arr, Y_arr, cand, length_scale, sigma_f, noise)
                ei = expected_improvement(mu, sigma, best_f, xi=0.01)
                # Add small random noise to break ties
                ei += rng.random(len(ei)) * 1e-6
                idx = np.argmax(ei)
                x_new = cand[idx]

            # Adaptive length scale: shrink as we converge
            if best_f < 10.0:
                length_scale = max(0.5, length_scale * 0.95)

            x_new = np.clip(x_new, lo, hi)
            f_new = func(x_new)
            calls_made += 1
            X.append(x_new)
            Y.append(f_new)
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            # Check if we should switch to local refinement
            # If best_f is reasonably low and we've made some progress
            if best_f < 5.0 and calls_made > 60 and not use_local_refine:
                use_local_refine = True

            # Switch to local refinement if we've been stuck
            if use_local_refine and calls_made > 100:
                # Do a local Nelder-Mead refinement using func
                try:
                    res = minimize(
                        lambda v: func(np.clip(v, lo, hi)),
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6},
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, lo, hi)
                        # Record this evaluation
                        X.append(best_x)
                        Y.append(best_f)
                        calls_made += 1
                except Exception:
                    pass

                # After local refinement, continue with surrogate but tighter
                length_scale = max(0.3, length_scale * 0.8)

        return best_x, best_f
# EVOLVE-BLOCK-END
