"""CMA-ES optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 30 and adaptive stopping."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize population size
        lambda_ = 30
        mu = lambda_ // 2  # number of parents

        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial sigma
        sigma = 2.0

        # Initial mean - start at origin which is often good for these functions
        xmean = np.zeros(dim)
        fx_mean = func(xmean)

        # PCA
        C = np.eye(dim)
        Csqrt = np.eye(dim)
        Cinv = np.eye(dim)
        Cinv_sqrt = np.eye(dim)

        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Best solution tracking
        best_x = xmean.copy()
        best_fx = fx_mean

        # Stopping criteria
        max_iterations = budget // lambda_  # roughly how many generations we can afford
        # Actually, let's be more careful. We have budget evaluations total.
        # First eval is xmean. Then each generation uses lambda_ evals.
        # So max gens = (budget - 1) // lambda_
        max_gens = (budget - 1) // lambda_

        for gen in range(max_gens):
            # Sample lambda_ points
            pop = np.zeros((lambda_, dim))
            pop_fx = np.zeros(lambda_)

            for i in range(lambda_):
                z = rng.standard_normal(dim)
                pop[i] = xmean + sigma * (Csqrt @ z)
                # Clip to box
                pop[i] = np.clip(pop[i], -5, 5)
                pop_fx[i] = func(pop[i])

            # Sort by fitness
            order = np.argsort(pop_fx)
            pop_sorted = pop[order]
            pop_fx_sorted = pop_fx[order]

            # Update best
            if pop_fx_sorted[0] < best_fx:
                best_fx = pop_fx_sorted[0]
                best_x = pop_sorted[0].copy()

            # Recombination
            xnew = np.dot(weights, pop_sorted[:mu])

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_sqrt @ ((xnew - xmean) / sigma))
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1)) / np.sqrt(mueff)
            pc = (1 - cc) * pc + hsig < 1.4 + 2 / (dim + 1) and cc / (1 - cc) * (xnew - xmean) / sigma or (1 - cc) * pc

            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig * np.outer(ps, ps) - c1) * C + \
                cmu * np.sum([np.outer(pop_sorted[i] - xmean, pop_sorted[i] - xmean) for i in range(mu)], axis=0) / (sigma ** 2)

            # Ensure symmetry
            C = 0.5 * (C + C.T)

            # Eigen-decomposition for square root and inverse square root
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues to be positive
                eigvals = np.maximum(eigvals, 1e-12)
                Csqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                Cinv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                # Fallback if decomposition fails
                Csqrt = np.eye(dim)
                Cinv_sqrt = np.eye(dim)
                Cinv = np.eye(dim)

            # Update mean
            xmean = xnew

            # Adapt sigma
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / np.sqrt(mueff) - 1))

            # Stopping criteria
            if sigma < 1e-12:
                break
            if np.max(np.abs(xmean)) > 1e6:
                break

        return best_x, best_fx


# Helper variables that were referenced but need to be defined in __call__
# Actually, I need to move the cs and dsigma definitions into __call__
# Let me rewrite the class properly
# EVOLVE-BLOCK-END
