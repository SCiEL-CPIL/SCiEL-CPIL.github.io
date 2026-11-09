"""CMA-ES optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation tuned for 5D BBOB f7 with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters for dim=5
        # Population size: lambda = 4 + floor(3 * log(dim))
        lam = max(4, 4 + int(3 * np.log(dim)))
        # For dim=5: lam = 4 + floor(3*1.609) = 4 + 4 = 8

        # Initialize at origin (shift is hidden but minimizer is in [-4,4]^5)
        # Try a few random starting points to avoid bad local minima
        best_x = None
        best_f = np.inf

        # Phase 1: Quick random search to find a good starting point
        n_random = min(20, budget // 10)
        for i in range(n_random):
            if time.time() > deadline - 5:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES from the best starting point
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # CMA-ES initialization
        x_mean = best_x.copy()
        sigma = 1.0  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)
        B = np.eye(dim)  # Eigenvectors
        D = np.ones(dim)  # Eigenvalues (square roots)
        c_sigma = (mu_eff + 2) / ((dim + 1) ** 1.5 + mu_eff)
        c_c = (4 + dim / c_sigma) / ((dim + 2) ** 1.5 + mu_eff)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        mu_eff = lam / 2  # Effective population size
        mu = [max(0.0, 0.5 * np.log(lam / 2 - i)) for i in range(lam)]
        mu = np.array(mu)
        w = mu / mu.sum()

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls_made = n_random

        # Main CMA-ES loop
        while calls_made < budget:
            if time.time() > deadline - 2:
                break

            # Generate offspring
            W = np.array([B @ (D * rng.standard_normal(dim)) for _ in range(lam)])
            Y = x_mean[None, :] + sigma * W
            Y = np.clip(Y, -5, 5)

            # Evaluate offspring
            F = []
            for y in Y:
                fy = func(y)
                F.append(fy)
                calls_made += 1
                if time.time() > deadline - 2:
                    break

            if len(F) < lam:
                # Pad with worst value if we ran out of time
                F.extend([best_f] * (lam - len(F)))

            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = np.array(F)[idx]

            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()

            # Compute weighted mean
            x_new = np.sum(w[:, None] * Y[:int(mu_eff)], axis=0)

            # Update evolution paths
            diff = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv @ diff

            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - n_random)))) / chi_N < 1.4 + 2 / (dim + 1) else 0.0

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_h * C) + c_mu * np.sum(w[:, None, None] * np.outer(Y[:int(mu_eff)] - x_mean[None, :], Y[:int(mu_eff)] - x_mean[None, :]), axis=0)

            # Update step size
            sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / chi_N - 1))

            # Update mean
            x_mean = x_new

            # Re-estimate covariance matrix eigen decomposition periodically
            if calls_made % lam == 0:
                try:
                    D_sq, B = np.linalg.eigh(C)
                    D_sq = np.clip(D_sq, 1e-20, 1e10)
                    D = np.sqrt(D_sq)
                    C_inv = (B @ np.diag(1.0 / D_sq)) @ B.T
                except np.linalg.LinAlgError:
                    pass

            # Check convergence
            if sigma < 1e-10 or np.all(np.abs(x_mean - best_x) < 1e-8):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
