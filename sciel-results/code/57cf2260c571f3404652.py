"""Optimized program for BBOB f21 (Gallagher 101 peaks) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based CMA-ES for multimodal continuous optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 1: Global scan to find a good starting point ---
        # Use a larger fraction of budget for global search on multimodal f21
        n_scan = min(100, budget // 5)
        if n_scan < 20:
            n_scan = 20

        # Generate initial points: mix of Latin hypercube and random
        points = []
        # Latin hypercube-like sampling with stratification
        for i in range(n_scan // 2):
            x = np.zeros(dim)
            for d in range(dim):
                x[d] = ((i + (d + 0.5) / (n_scan // 2)) % 1.0) * 10.0 - 5.0
            perm = rng.permutation(dim)
            x = x[perm]
            points.append(x)

        # Additional random points for diversity
        for i in range(n_scan - n_scan // 2):
            x = rng.uniform(-5, 5, dim)
            points.append(x)

        # Evaluate all scan points
        best_x = None
        best_f = np.inf
        for x in points:
            fx = func(np.clip(x, -5, 5))
            if fx < best_f:
                best_f = fx
                best_x = np.clip(x, -5, 5)

        # --- Phase 2: CMA-ES refinement ---
        # Remaining budget for CMA-ES
        remaining = budget - n_scan
        if remaining <= 0:
            return best_x, best_f

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * ln(dim))
        lam = max(4 + int(3 * np.log(dim)), 12)
        # Ensure lam is not too large for remaining budget
        max_iter = remaining // lam
        if max_iter < 5:
            lam = max(4, remaining // 5)
            max_iter = remaining // lam

        # Initialize mean at best scan point
        mean = best_x.copy()
        # Large initial sigma to explore multiple basins
        sigma = 3.0

        # CMA-ES weights and parameters
        mu = lam // 2  # number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Covariance matrix (start as identity)
        C = np.eye(dim)
        # Eigen decomposition cache
        eigvals, eigvecs = np.linalg.eigh(C)

        # Helper: update eigen decomposition if needed
        def get_eig():
            nonlocal eigvals, eigvecs
            # Recompute if C has changed significantly (simplified: always recompute for correctness)
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            return eigvals, eigvecs

        # Helper: matrix square root via eigen decomposition
        def matrix_sqrt(C, eigvals, eigvecs):
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        # Helper: matrix inverse via eigen decomposition
        def matrix_inv(C, eigvals, eigvecs):
            return eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

        # Helper: matrix product via eigen decomposition
        def matrix_prod(A, B, eigvals, eigvecs):
            # A @ B where A is symmetric and we have its eigendecomp
            return eigvecs @ (np.diag(eigvals) @ (eigvecs.T @ B))

        best_x_cma = best_x.copy()
        best_f_cma = best_f

        calls_made = n_scan
        for it in range(max_iter):
            if calls_made >= budget:
                break

            # Get current eigen decomposition
            eigvals, eigvecs = get_eig()
            B = eigvecs
            D = np.sqrt(eigvals)
            # C^(1/2) = B * diag(sqrt(D))
            B_d = B * D  # each column scaled

            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = B_d @ z
                x_new = np.clip(mean + sigma * y, -5, 5)
                offspring.append(x_new)

            # Evaluate offspring
            fitnesses = []
            for x_new in offspring:
                if calls_made >= budget:
                    # Fill remaining with inf
                    fitnesses.append(np.inf)
                else:
                    f_val = func(x_new)
                    fitnesses.append(f_val)
                    calls_made += 1

            fitnesses = np.array(fitnesses)
            # Sort by fitness
            order = np.argsort(fitnesses)
            y_sorted = [offspring[i] for i in order]
            f_sorted = fitnesses[order]

            # Update best
            if f_sorted[0] < best_f_cma:
                best_f_cma = f_sorted[0]
                best_x_cma = y_sorted[0].copy()

            # Update mean
            y_weighted = np.zeros(dim)
            for i in range(mu):
                y_weighted += weights[i] * (y_sorted[i] - mean) / sigma
            mean_old = mean.copy()
            mean = mean + sigma * y_weighted
            mean = np.clip(mean, -5, 5)

            # Update sigma (CSA)
            threshold = (1 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)) * 1.4
            h_sigma = 1.0 if np.linalg.norm(y_weighted) / np.sqrt(mu_eff) < threshold else 0.0

            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_weighted
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update C (rank-mu + rank-1 update)
            # Rank-1
            y_norm = (mean - mean_old) / sigma
            rank1 = np.outer(y_norm, y_norm)

            # Rank-mu
            y_mu = np.zeros((dim, mu))
            for i in range(mu):
                y_mu[:, i] = (y_sorted[i] - mean_old) / sigma
            rank_mu = np.outer(y_mu, weights[None, :] * y_mu)

            C = (1 + c_1 * (c_mu - c_1) + c_mu) * C \
                - c_1 * rank1 \
                - c_mu * rank_mu

            # Ensure positive definiteness
            eigvals_new, eigvecs_new = np.linalg.eigh(C)
            eigvals_new = np.maximum(eigvals_new, 1e-12)
            C = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T
            eigvals, eigvecs = eigvals_new, eigvecs_new

        return best_x_cma, best_f_cma
# EVOLVE-BLOCK-END
