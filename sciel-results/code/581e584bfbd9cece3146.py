"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Initial center: origin is a reasonable guess for shifted/rotated landscape
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(n))  # ~11 for n=5
        # Weighting
        mu = lam // 2
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size
        sigma = 2.0

        # Covariance matrix
        C = np.eye(n)
        Cs = np.eye(n)  # evolution path for covariance

        # Evolution path for step size
        ps = np.zeros(n)
        cs = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        ds = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + cs

        # Eigen decomposition cache
        eigvals, eigvecs = np.linalg.eigh(C)

        # Count function evaluations
        eval_count = 1

        # Main loop
        while eval_count < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Update eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
            except:
                eigvals = np.ones(n) * 1e-10
                eigvecs = np.eye(n)

            # Generate population
            Y = np.zeros((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                # Transform using eigen decomposition
                y = eigvecs @ (np.sqrt(eigvals) * z)
                Y[i] = np.clip(x + sigma * y, -5, 5)

            # Evaluate population
            F = np.array([func(Y[i]) for i in range(lam)])
            eval_count += lam

            # Update best
            if F.min() < best_f:
                best_f = F.min()
                best_x = Y[np.argmin(F)].copy()

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Weighted mean of best mu individuals
            m = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

            # Evolution path for step size
            inv_sigma = 1.0 / sigma if sigma > 1e-30 else 1.0
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * inv_sigma * (eigvecs @ (np.sqrt(eigvals) * (eigvecs.T @ (m - x))))

            # Update sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update mean
            x = m

            # Evolution path for covariance
            Cs = (1 - cs) * Cs + np.sqrt(cs * (2 - cs) * mu_eff) * inv_sigma * (eigvecs @ (np.sqrt(eigvals) * (eigvecs.T @ (m - x))))

            # Update covariance matrix
            delta_sigma = (1 + 2 * max(0, (np.linalg.norm(ps) / chi_n - 1)) * cs) * ds
            C = (1 - cs * (2 - cs) - cs * delta_sigma * mu_eff) * C \
                + cs * delta_sigma * np.outer(Cs, Cs) \
                + mu_eff * np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu] - x, Y_sorted[:mu] - x), axis=0)

            # Ensure positive definiteness
            eigvals_new, _ = np.linalg.eigh(C)
            if np.any(eigvals_new < 1e-20):
                C = C + 1e-10 * np.eye(n)

        return best_x, best_f
# EVOLVE-BLOCK-END
