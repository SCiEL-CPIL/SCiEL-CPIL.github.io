"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic CMA-ES variant with Latin Hypercube initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        n_calls = 0

        # Latin Hypercube initialization: sample several points for better initial coverage
        # Use 4 LHS points to get a diverse starting set
        n_init = min(4, budget)
        rng_init = np.random.default_rng(12345)
        
        # Generate LHS points
        init_points = []
        for d in range(dim):
            # For each dimension, create a permutation and add jitter
            perm = rng_init.permutation(n_init)
            for i in range(n_init):
                if i == 0:
                    pt = np.zeros(dim)
                pt[d] = -5 + 10 * (perm[i] + rng_init.uniform(0, 1)) / n_init
                if i == 0:
                    init_points.append(pt.copy())
        
        # Evaluate all init points and pick the best
        best_x = None
        best_f = np.inf
        for pt in init_points:
            pt = np.clip(pt, -5, 5)
            f_val = func(pt)
            n_calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()
        
        x = best_x
        fx = best_f

        # CMA-ES parameters
        # Use population size tuned for dim=5
        lam = 10
        mu = lam // 2

        # Step size - start with a reasonable scale for [-5,5]^5
        sigma = 2.0

        # Covariance matrix (identity initially)
        C = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1. / np.sum(weights**2)

        # Learning rates
        c_sigma = (mu_eff + 0.5) / (dim + 2)
        c_c = (4 - mu_eff + c_sigma) / ((dim + 1.3) * (dim + mu_eff))
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2.)**2 + mu_eff))

        # Damping
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1 / (21 * dim**2))

        # Main CMA-ES loop
        gen = 0
        while n_calls < budget:
            # Eigendecomposition of C
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-14)
            B = evecs
            D = np.sqrt(evals)

            # Sample population
            # Use deterministic seed based on generation for reproducibility
            rng = np.random.default_rng(42 + gen)
            Y = rng.standard_normal((lam, dim))

            # Transform to get X
            X = np.tile(x, (lam, 1)) + sigma * (Y @ (B * D))

            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate population
            F = np.empty(lam)
            for i in range(lam):
                if n_calls >= budget:
                    break
                F[i] = func(X[i])
                n_calls += 1

            # If we ran out of budget, update best and break
            if n_calls >= budget:
                for i in range(lam):
                    if F[i] < fx:
                        fx = F[i]
                        x = X[i].copy()
                break

            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]

            # Update best
            if F_sorted[0] < fx:
                fx = F_sorted[0]
                x = X_sorted[0].copy()

            # Recombination
            m = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)

            # Update evolution path for sigma
            z = (m - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D @ z))

            # Hessian update
            norm_p_sigma = np.linalg.norm(p_sigma)
            threshold = (1.4 + 2 / (dim + 1)) * chi_n
            h_sigma = 1 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(gen + 1)) < threshold else 0

            # Update evolution path for C
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (D @ z))

            # Update covariance matrix using rank-one and rank-mu updates
            delta_h = (h_sigma - 1) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + \
                c_1 * np.outer(p_c, p_c) + \
                c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(X_sorted[:mu] - x, X_sorted[:mu] - x) / sigma**2, axis=0)

            # Symmetrize for numerical stability
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))

            # Clamp sigma to prevent numerical issues
            sigma = np.clip(sigma, 1e-10, 1e4)

            # Update mean
            x = m

            gen += 1

        return x, fx
# EVOLVE-BLOCK-END
