"""Two-phase hybrid optimizer: Sobol exploration + CMA-ES refinement for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: broad quasi-random exploration then CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ---- Phase 1: Sobol quasi-random exploration ----
        # Use a low-discrepancy sequence to spread evaluations across [-5,5]^dim
        # This robustly finds the shifted/rotated basin regardless of hidden transforms.
        n_explore = min(150, budget)  # 150 points for exploration

        # Generate Sobol-like quasi-random points using inverse CDF of uniform
        # with a scrambled Halton-style sequence for good space-filling in 5D
        # We'll use a simple approach: stratified sampling per dimension
        # For 150 points in 5D, we can do a 5D grid of ~2.7 per axis, but that's
        # too coarse. Instead, use a scrambled Halton sequence.

        def halton_sequence(n, dim):
            """Generate n Halton quasi-random points in [0,1)^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # bases: 2, 3, 5, 7, 11, ...
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    idx = i + 1
                    while idx > 0:
                        f /= base
                        r += f * (idx % base)
                        idx //= base
                    points[i, d] = r
            # Scramble: add a random shift mod 1 to break correlations
            shift = np.random.rand(dim)
            points = (points + shift) % 1.0
            return points

        # Map [0,1)^dim to [-5, 5]^dim
        sobol_pts = halton_sequence(n_explore, dim)
        sobol_pts = sobol_pts * 10.0 - 5.0

        for pt in sobol_pts:
            if total_calls >= budget or time.time() > deadline:
                break
            safe_call(pt)

        if total_calls >= budget or time.time() > deadline:
            return best_x, best_f

        # ---- Phase 2: CMA-ES refinement from best point ----
        # Initialize CMA-ES at the best point found, with a moderate step size
        # to exploit the local structure of the ill-conditioned landscape.
        lam = 14
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)

        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        c1 = (2 - 1 / mu_eff + 2 * mu_eff / dim) / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        mean = best_x.copy()
        # Start with a step size that covers the likely local basin
        # For Schaffers F7, the basin is relatively tight around the minimizer
        sigma = 1.5

        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        def update_cholesky(C_mat):
            try:
                L = np.linalg.cholesky(C_mat + 1e-12 * np.eye(dim))
                return L
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-12, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                return L

        L = update_cholesky(C)
        chi_lam = np.sqrt(lam) * (1 - 1 / (2 * lam**2) + 1 / (8 * lam**4))

        while total_calls < budget and time.time() < deadline:
            # Generate population
            zs = np.random.randn(lam, dim)
            ys = (L @ zs.T).T * sigma + mean

            # Evaluate population
            fs = np.array([safe_call(y) for y in ys])

            if total_calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Update mean
            old_mean = mean.copy()
            mean = np.sum(ys[:mu] * w[:, np.newaxis], axis=0)

            # Update evolution path for sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ (mean - old_mean) / sigma)
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1 if norm_p_sigma / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (dim + 1)) * chi_lam else 0

            # Update evolution path for C
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma

            # Update C
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * cc * (2 - cc) * C)

            diff = (ys[:mu] - old_mean) / sigma
            term = diff.T @ (w[:, np.newaxis] * diff)
            C += cmu * term
            C = (C + C.T) / 2.0

            L = update_cholesky(C)

            # Update C_inv_sqrt via eigen decomposition (cheap for dim=5)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Adapt sigma
            sigma *= np.exp(cs / damps * (norm_p_sigma / chi_lam - 1))

            if sigma > 50:
                sigma = 50
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
