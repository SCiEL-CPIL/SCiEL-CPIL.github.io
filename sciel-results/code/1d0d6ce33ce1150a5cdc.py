"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s kill

        # Initialize
        x0 = np.zeros(dim)
        sigma = 2.0

        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = max(1, lam // 2)

        # CMA-ES parameters
        cm = 1.0 / (dim + 4.0)
        cc = (dim + 1.5) / (dim + 9.0)
        cs = (cc + 2) / (dim + cc + 5)
        ds = 1 + 2 * max(0, np.sqrt((cs + 1) / (cs + 3) - 1)) + cs

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        c1 = (2 * (mueff - 2 + 2 / dim)) / ((dim + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / dim) / ((dim + 2) ** 2 + mueff))

        # State
        mean = x0.copy()
        C = np.eye(dim)
        Cinv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cholesky_inv_sqrt(Cmat):
            """Compute C^{-1/2} via Cholesky."""
            L = np.linalg.cholesky(Cmat)
            # C = L @ L.T, so C^{-1/2} = (L^{-1})^T @ L^{-1}
            Linv = np.linalg.inv(L)
            return Linv.T @ Linv

        # Main loop
        eval_count = 0
        best_x = mean.copy()
        best_f = float('inf')

        while eval_count < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Compute C^{-1/2}
            try:
                Cinv_sqrt = cholesky_inv_sqrt(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, use pseudo-inverse
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-15, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except:
                    Cinv_sqrt = np.eye(dim)

            # Sample population
            Y = np.random.randn(lam, dim)
            X = np.array([mean + sigma * (Cinv_sqrt @ y.T).T for y in Y])
            # Clip to bounds
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(x) for x in X])
            eval_count += lam

            # Update best
            for i in range(lam):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()

            # Sort by fitness
            order = np.argsort(F)
            Xs = X[order]
            Fs = F[order]

            # Weighted mean
            m_new = np.sum(weights[:, None] * Xs[:mu], axis=0)

            # Update paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cinv_sqrt @ (m_new - mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (eval_count + 1)) / chiN) < (1.4 + 2 / (dim + 1))
            pc = (1 - cm) * pc + hsig * cm * (m_new - mean) / sigma

            # Update covariance
            delta_h = (1 - hsig) * cc * (1 + cc - cm)
            C = ((1 + c1 * cmu - c1) * C +
                 c1 * (np.outer(pc, pc) + delta_h * C - np.eye(dim)) +
                 cmu * np.sum(weights[:, None, None] * np.outer(Xs[:mu] - mean, Xs[:mu] - mean), axis=0))

            # Update mean and sigma
            mean = m_new
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
            sigma = max(sigma, 1e-12)

            # Check convergence
            if np.linalg.norm(C - np.trace(C) / dim * np.eye(dim)) / max(1, np.trace(C) / dim) < 1e-10:
                break

            # If sigma is too small, restart with larger sigma
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
