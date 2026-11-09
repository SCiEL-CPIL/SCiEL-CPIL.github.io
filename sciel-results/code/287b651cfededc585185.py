"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f18 with PCA-initialized covariance."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = max(6, int(4 + 3 * np.log(dim)))
        mu = lam // 2

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim * dim) + 1 / (12 * dim ** 3))

        # Phase 1: Initial sampling to estimate covariance structure
        n_init = min(30, budget - lam)
        if n_init < 2:
            n_init = 2

        init_samples = []
        init_fits = []
        calls = 0

        # Sample initial points spread over the domain
        for i in range(n_init):
            x_init = rng.uniform(-4, 4, size=dim)
            f_init = func(x_init)
            init_samples.append(x_init)
            init_fits.append(f_init)
            calls += 1
            if calls >= budget:
                break

        init_samples = np.array(init_samples[:calls])
        init_fits = np.array(init_fits[:calls])

        # Use the best few points to estimate the search direction
        n_best_init = max(3, min(8, len(init_samples)))
        idx_best = np.argsort(init_fits)[:n_best_init]
        best_init_samples = init_samples[idx_best]
        mean_init = best_init_samples.mean(axis=0)

        # Estimate covariance from initial samples (weighted by fitness)
        # Use all initial samples for a broader view
        if len(init_samples) >= 2:
            centered = init_samples - init_samples.mean(axis=0)
            # Weight by inverse fitness (better points get more weight)
            f_min = init_fits.min()
            f_max = init_fits.max()
            if f_max > f_min:
                w_init = 1.0 / (1.0 + (init_fits - f_min) / (f_max - f_min + 1e-10))
            else:
                w_init = np.ones(len(init_samples))
            w_init /= w_init.sum()

            # Weighted covariance
            C_init = np.zeros((dim, dim))
            for i in range(len(init_samples)):
                C_init += w_init[i] * np.outer(centered[i], centered[i])
            C_init /= (1 - (w_init ** 2).sum()) if (1 - (w_init ** 2).sum()) > 1e-10 else 1.0
        else:
            C_init = np.eye(dim)

        # Ensure C_init is positive definite
        eigvals, eigvecs = np.linalg.eigh(C_init)
        eigvals = np.clip(eigvals, 1e-8, None)
        C_init = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Initialize CMA-ES state
        x = mean_init.copy()
        sigma = 1.0
        C = C_init.copy()
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Get best from initial phase
        best_x = init_samples[idx_best[0]].copy()
        best_f = init_fits[idx_best[0]]

        # Get initial mean fitness for scaling
        init_mean_f = init_fits.mean()
        init_std_f = init_fits.std() + 1e-10

        # Phase 2: CMA-ES
        while calls < budget:
            # Sample population
            ys = np.zeros((lam, dim))
            fs = np.zeros(lam)

            for i in range(lam):
                z = rng.standard_normal(dim)
                # Transform using Cholesky of C
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    y = x + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    y = x + sigma * z

                ys[i] = np.clip(y, -5, 5)
                fs[i] = func(ys[i])
                calls += 1
                if calls >= budget:
                    break

            if calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]

            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()

            # Weighted mean of selected points
            y_w = np.sum(weights[:, None] * (ys_sorted[:mu] - x), axis=0)

            # Update evolution path ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w

            # h_sigma: check if ps is within expected range
            h_sigma = 1.0
            if sigma > 1e-10:
                ps_norm = np.linalg.norm(ps)
                # Expected norm of ps under null hypothesis
                exp_norm = chi_n * np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, calls - n_init)))
                if exp_norm > 1e-10:
                    h_sigma = 1.0 if ps_norm / exp_norm < 1.4 * (1 + 2 / (dim + 1)) else 0.0

            # Update evolution path pc
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + \
                c_1 * np.outer(pc, pc) + \
                np.sum([weights[i] * np.outer(ys_sorted[i] - x, ys_sorted[i] - x) for i in range(mu)], axis=0) * c_mu

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update step size
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = x + y_w

            # Optional: adaptive sigma based on fitness improvement
            if len(fs_sorted) >= 2:
                improvement = init_mean_f - fs_sorted[0]
                if improvement > 0 and sigma < 5.0:
                    # Slight increase in exploration when making progress
                    pass  # Keep standard CMA-ES step size control

        return best_x, best_f
# EVOLVE-BLOCK-END
