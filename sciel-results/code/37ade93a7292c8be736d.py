"""Rewritten optimizer using CMA-ES style adaptive covariance evolution."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for box-constrained optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        # Weighting
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step-size control parameters
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        # Covariance matrix adaptation
        c1 = (2 * (mueff - 2 + 2 / dim)) / ((dim + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # Damping for step size
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialization
        # Start near center, but let the algorithm adapt
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        Cinv = np.eye(dim)
        # Precompute Cholesky for sampling
        Csqrt = np.eye(dim)
        Csqrinv = np.eye(dim)
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Best solution tracking
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        # Helper to update Cholesky factors approximately
        def update_cholesky(C, Cinv):
            # For stability, we recompute Cholesky periodically or use stable updates
            # Here we just recompute for simplicity, as dim is small (5)
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                Csqrt = L
                # Inverse Cholesky
                L_inv = np.linalg.inv(L.T)
                Csqrinv = L_inv
                Cinv = L_inv.T @ L_inv
            except np.linalg.LinAlgError:
                # Fallback if singular
                C = C + 1e-12 * np.eye(dim)
                L = np.linalg.cholesky(C)
                Csqrt = L
                L_inv = np.linalg.inv(L.T)
                Csqrinv = L_inv
                Cinv = L_inv.T @ L_inv
            return C, Cinv, Csqrt, Csqrinv

        C, Cinv, Csqrt, Csqrinv = update_cholesky(C, Cinv)

        for _ in range(budget - 1):
            # Generate population
            ys = np.empty((lambda_, dim))
            fs = np.empty(lambda_)
            
            for i in range(lambda_):
                # Sample from N(0, C)
                z = rng.standard_normal(dim)
                y = x + sigma * (Csqrt @ z)
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls += 1
                if calls > budget:
                    break
            
            if calls > budget:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()

            # Mean update
            y_w = np.sum(weights[:, np.newaxis] * ys[:mu], axis=0)
            hsig = 1.0 if np.sqrt((1 - (1 - sigma) ** (2 * (budget - calls))) / (1 - (1 - sigma) ** 2)) < (1 + cs) * chi_n else 0.0
            # Simpler hsig check for stability
            p_sigma = (1 - cs) * p_sigma + hsig * np.sqrt(cs * (2 - cs) * mueff) * (Cinv @ (sigma * y_w))
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Update C
            p_c = (1 - c1) * p_c + hsig * np.sqrt(c1 * (2 - c1) * mueff) * y_w
            delta_hsig = (1 - (1 - cs) ** 2) * 1 + cs * (2 - cs) * mueff
            C = (1 + c1 * (mueff - 1) - c1) * C \
                + c1 * (np.outer(p_c, p_c) - (1 - (1 - c1) ** 2) * C) \
                + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(ys[:mu] - y_w, ys[:mu] - y_w), axis=0)
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            C, Cinv, Csqrt, Csqrinv = update_cholesky(C, Cinv)

            # Update mean
            x = x + y_w
            x = np.clip(x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
