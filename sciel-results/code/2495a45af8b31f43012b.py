"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with PCA-initialized covariance for Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin

        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2

        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # CMA-ES parameters
        c_sigma = (mueff + 2) / ((dim + 1.3) ** 2 + mueff)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / dim) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial point: center of the box
        x_mean = np.zeros(dim)
        sigma = 2.0

        # PCA initialization: sample a few points to estimate the landscape anisotropy
        # For Schaffers F7, the ill-conditioning creates a funnel structure.
        # We use a small set of evaluations to build an initial covariance.
        # Start with identity, then refine with a few directional probes.
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Evaluate initial point
        fx_best = func(x_mean.copy())
        best_x = x_mean.copy()
        best_f = fx_best
        calls = 1

        # PCA initialization: probe along random directions to estimate anisotropy
        rng = np.random.default_rng(42)
        n_probes = min(2 * dim, 10)
        probe_points = []
        probe_vals = []
        for _ in range(n_probes):
            d = rng.standard_normal(dim)
            d /= np.linalg.norm(d)
            p = x_mean + sigma * d
            p = np.clip(p, -5, 5)
            v = func(p)
            calls += 1
            probe_points.append(p - x_mean)
            probe_vals.append(v - fx_best)

        # Estimate covariance from probes: points with lower values are more promising
        # Use inverse-variance weighting
        probe_arr = np.array(probe_points)  # (n_probes, dim)
        probe_vals_arr = np.array(probe_vals)  # (n_probes,)

        # Weight by how good the probe was (lower value = higher weight)
        # Clip to avoid division by zero
        weights_p = 1.0 / (np.abs(probe_vals_arr) + 1e-6)
        weights_p /= weights_p.sum()

        # Weighted covariance
        centered = probe_arr - np.sum(weights_p[:, None] * probe_arr, axis=0, keepdims=True)
        C_est = (centered.T * weights_p) @ centered / (np.sum(weights_p) + 1e-10)
        C_est += 1e-6 * np.eye(dim)  # Regularize

        # Check if C_est is positive definite
        try:
            eigvals, eigvecs = np.linalg.eigh(C_est)
            if np.all(eigvals > 1e-8):
                C = C_est
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
                C_sqrt = np.linalg.cholesky(C)
            else:
                # Fallback to identity
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                C_sqrt = np.eye(dim)
        except np.linalg.LinAlgError:
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            C_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Main CMA-ES loop
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            z = rng.standard_normal((lam, dim))
            # Transform: x_i = x_mean + sigma * C_sqrt @ z_i
            pops = x_mean[None, :] + sigma * (C_sqrt @ z.T).T
            pops = np.clip(pops, -5, 5)

            # Evaluate population
            fvals = np.zeros(lam)
            for i in range(lam):
                fvals[i] = func(pops[i])
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pops[i].copy()

                # Check deadline inside loop
                if time.time() > deadline:
                    break

            # If we broke early due to deadline, pad fvals
            if calls >= budget or time.time() > deadline:
                if calls < calls + lam:
                    # We didn't finish the population; break out
                    break

            # Sort by fitness
            order = np.argsort(fvals)
            sorted_pops = pops[order]
            sorted_fvals = fvals[order]

            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * sorted_pops[:mu], axis=0)
            x_mean = np.clip(x_mean, -5, 5)

            # Update sigma (CSA)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                C_inv_sqrt @ (x_mean - x_old) / sigma
            )
            h_sigma = (
                np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls)))
                < (1 + 2 / (dim + 1)) * chi_n
            )
            if h_sigma:
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update C
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (
                x_mean - x_old
            ) / sigma
            h_c = (
                np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (budget - calls)))
                < (1 + 2 / (dim + 1)) * chi_n
            )

            # Rank-mu update
            delta_h = (1 - h_c) * c_c * (2 - c_c)
            C = (
                (1 + c_1 + c_mu - c_1 - c_mu + delta_h) * C
                + c_1 * np.outer(p_c, p_c)
                + delta_h * np.outer(p_c, p_c)
                + c_mu / sigma ** 2 * np.sum(weights[:mu][:, None, None] * np.outer(sorted_pops[:mu] - x_old, sorted_pops[:mu] - x_old), axis=0)
            )

            # Ensure positive definiteness
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = (eigvecs * eigvals) @ eigvecs.T
                C = (C + C.T) / 2
            except np.linalg.LinAlgError:
                C = np.eye(dim)

            # Update C_inv_sqrt and C_sqrt
            try:
                L = np.linalg.cholesky(C)
                C_sqrt = L
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = (eigvecs * eigvals) @ eigvecs.T
                C = (C + C.T) / 2
                L = np.linalg.cholesky(C)
                C_sqrt = L
                C_inv_sqrt = np.linalg.inv(L)

            # Stop if sigma is too small
            if sigma < 1e-12:
                break

            # Check time
            if time.time() > deadline - 5:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
