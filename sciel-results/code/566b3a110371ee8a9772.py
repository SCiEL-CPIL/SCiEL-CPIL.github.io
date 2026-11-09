"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        n = self.dim
        # Population size: lambda = 4 + 3*log(n)
        lam = int(4 + 3 * np.log(n))
        lam = max(lam, 8)

        # Initial mean and sigma
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-4, 4, n)
        sigma = 1.5

        # CMA-ES parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Track best
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1

        # Main loop
        while calls_made < self.budget:
            if time.time() > deadline:
                break

            # Generate population
            ps = np.zeros((lam, n))
            ys = np.zeros((lam, n))
            fs = np.zeros(lam)

            for i in range(lam):
                z = rng.standard_normal(n)
                ps[i] = z
                y = np.clip(x_mean + sigma * (C @ z), -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls_made += 1
                if calls_made >= self.budget or time.time() > deadline:
                    break

            if calls_made >= self.budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            ps = ps[order]
            fs = fs[order]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()

            # Compute new mean
            x_new = sum(weights[i] * ys[i] for i in range(mu))

            # Update step-size control
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ (sigma * (x_new - x_mean) / sigma))
            # Actually, the standard formula: p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * C^(-1/2) * (x_new - x_mean)/sigma
            # But C^(-1/2) is expensive. Let's use the standard CMA-ES update.
            
            # Recompute p_sigma properly
            # The standard CMA-ES update for p_sigma involves C^{-1/2}
            # For simplicity and speed, let's use a simplified version or the standard one
            # Standard: p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * C^{-1/2} * (x_new - x_mean)/sigma
            
            # For now, let's use the standard CMA-ES update
            # We need C^{-1/2}. We can compute it via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_new - x_mean) / sigma)

            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, 0))) < (1 + 2 / (n + 1)) * chi_n else 0.0

            # Update translation
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + delta_h) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(ys[i] - x_mean, ys[i] - x_mean) for i in range(mu))

            # Symmetrize C
            C = (C + C.T) / 2

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Ensure sigma is within reasonable bounds
            sigma = np.clip(sigma, 1e-10, 50)

            # Update mean
            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
