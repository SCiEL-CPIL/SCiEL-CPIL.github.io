"""Model-and-optimize optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))
        sigma = 2.0
        # Covariance matrix (start as identity)
        C = np.eye(dim)
        # Eigen decomposition for sampling
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        # Step size control
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        # Path accumulators
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Adaptation rates
        cs = (lam/2 + 1) / (dim + lam/2 + 3)
        cmu = min(1 - 2*cs, (lam/2 - 2) / (dim + lam/2 + 3))
        # Update interval for covariance
        damps = 1 + 2*max(0, np.sqrt((lam/2 - 1)/(dim + 1)) - 1) + cs
        # Count for covariance update
        count = 0
        # Thresholds
        hsig = (1.4 + 2/(dim+1)) * (1 - cs)

        budget_left = self.budget - 1

        for _ in range(budget_left):
            # Deadline check
            if time.time() > deadline:
                break

            # Sample lambda offspring
            offspring = []
            for i in range(lam):
                # Sample z from standard normal
                z = rng.standard_normal(dim)
                # Transform using eigendecomposition
                # z -> eigenvectors @ (sqrt(eigenvalues) * z)
                y = x + sigma * (eigenvectors @ (np.sqrt(np.maximum(eigenvalues, 1e-12)) * z))
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate all offspring
            offspring_f = []
            for y in offspring:
                fy = func(y)
                offspring_f.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Sort by fitness
            indices = np.argsort(offspring_f)
            mu = max(1, lam // 2)
            weights = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)

            # Mean update
            x_old = x.copy()
            x = sum(weights[i] * offspring[indices[i]] for i in range(mu))

            # Update path covariance
            inv_sigma = 1.0 / sigma if sigma > 0 else 0
            # Need to compute (x - x_old) / sigma in the covariance frame
            # For simplicity, use the simple update
            d = (x - x_old) * inv_sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * d
            norm_ps = np.linalg.norm(ps)
            hsig = norm_ps < (1.4 + 2/(dim+1)) * chi_n

            # Update path evolution path
            pc = (1 - cmu) * pc + hsig * np.sqrt(cmu * (2 - cmu) * mueff) * d

            # Update covariance
            count += 1
            if count % max(1, int(1.0 / (cs * damps))) == 0 or count == 1:
                # Compute rank-one and rank-mu updates
                delta_hsig = (1 - hsig) * hsig * cs * (2 - cs)
                C = (1 + cs * damps * delta_hsig - cmu) * C
                C += cs * damps * delta_hsig * np.outer(pc, pc)
                for i in range(mu):
                    y_i = (offspring[indices[i]] - x_old) * inv_sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)

                # Re-orthogonalize: ensure C is symmetric PSD
                C = (C + C.T) / 2.0
                # Regularize
                C += 1e-12 * np.eye(dim)

                # Update eigendecomposition
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-12)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    eigenvalues = np.ones(dim)
                    eigenvectors = np.eye(dim)

            # Update sigma
            sigma *= np.exp(cs / damps * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
