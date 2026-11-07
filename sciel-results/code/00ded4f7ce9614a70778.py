"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start for Rastrigin on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Phase 1: Multi-start to find promising basins
        n_starts = 20
        candidates = []
        for _ in range(n_starts):
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            candidates.append((fx, x.copy()))
            if not time_ok():
                break

        candidates.sort(key=lambda t: t[0])
        # Take the best few starting points for CMA-ES

        best_x = candidates[0][1]
        best_fx = candidates[0][0]

        # Phase 2: CMA-ES on the best starting point
        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lam // 2  # number of parents
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = int(weights.sum() ** 2 / weights.sum())
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / dim) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize CMA-ES
        x_mean = best_x.copy()
        sigma = 1.0
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        evals = n_starts

        while evals < budget and time_ok():
            # Generate lambda offspring
            offspring = []
            for _ in range(lam):
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                offspring.append((fy, y.copy()))
                if not time_ok():
                    break

            if len(offspring) == 0:
                break

            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_fx:
                best_fx = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Select mu best
            selected = offspring[:mu]
            y_w = np.array([s[1] for s in selected])
            x_w = y_w.T @ weights  # weighted mean

            # Compute evolution path for sigma
            y_inv = np.zeros(dim)
            for i in range(mu):
                z_i = np.linalg.solve(B.T @ (D[:, None] * (D[:, None].T)), 
                                      (y_w[i] - x_mean) / sigma)
                y_inv += weights[i] * z_i

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_inv

            # Update mean
            x_mean_new = x_w.copy()

            # Evolution path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (
                (x_w - x_mean) / sigma
            )

            # Update covariance matrix
            outer = np.outer(x_w - x_mean, x_w - x_mean) / sigma**2
            C = (1 + c_1 * (mueff - 1) + c_mu) * (B @ (D**2) @ B.T) \
                - (1 + c_1 * (mueff - 1)) * (B @ (D**2) @ B.T) \
                + c_1 * (B @ (D**2) @ B.T) \
                + c_mu * np.sum(weights[:, None, None] * outer[:, None, :], axis=0)

            # More standard update:
            # C = (1 + c1*(mueff-1) + cmu - 1) * C_old + c1*(C_old - C) + cmu * sum(wi * (yi-xm)(yi-xm)^T)
            C_old = B @ (D**2) @ B.T
            C = (1 + c_1 * (mueff - 1) + c_mu) * C_old \
                - (1 + c_1 * (mueff - 1)) * C_old \
                + c_1 * (C_old - C_old) \
                + c_mu * np.sum(weights[:, None, None] * outer[:, None, :], axis=0)

            # Simpler: use the standard CMA-ES covariance update
            C = C_old + c_1 * (C_old - C_old) + c_mu * np.sum(
                weights[:, None, None] * outer[:, None, :], axis=0
            ) + c_1 * (np.outer(x_w - x_mean, x_w - x_mean) / sigma**2 - C_old)

            # Actually, let me use a cleaner implementation
            # Standard CMA-ES covariance update:
            # C <- (1 + c1*(mueff-1) + cmu) * C - c1*(C - C) + cmu * sum wi * (yi-xm)(yi-xm)^T/sigma^2
            # But the standard formula is:
            # C <- (1 - c1) * C + c1 * (C + outer) + cmu * sum...
            # Let me just do a simpler but effective update

            # Recompute B, D from C via eigendecomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                D = np.ones(dim)
                B = np.eye(dim)

            x_mean = x_mean_new

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

            # Check for convergence
            if sigma < 1e-6:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
