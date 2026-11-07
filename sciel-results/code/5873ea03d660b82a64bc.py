"""CMA-ES optimizer for the BBOB-style Rastrigin (f3) task."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with a large population for multimodal, rotated landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0   # safety margin before the 300s kill

        # Population size: larger lambda helps on multimodal Rastrigin.
        lam = 30
        mu = lam // 2                     # number of recombination parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - c_sigma - 2 / (dim + 1))
        c_mu = min(1 - c_1, (2 * (mu_eff / 2 - 1 + 2 / mu_eff)) *
                   (1 - c_sigma - c_1))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial step size: span the box to escape local minima early.
        sigma = 2.0

        # Initialize mean near the origin (shift is hidden but bounded in [-4,4]).
        m = np.zeros(dim)

        # Covariance matrix: identity (CMA-ES adapts it to the rotation).
        C = np.eye(dim)
        # Precompute C^{-1/2} (initially identity) for sampling.
        Cinv = np.eye(dim)
        # Eigen-decomposition cache for sampling and updates.
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-20, None)

        # Evolution paths.
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        d_amps = 1 + c_sigma + 2 / (dim - 1)

        def sample():
            """Draw a standard normal transformed by C^{-1/2}."""
            z = np.random.randn(dim)
            # C^{-1/2} = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs^T
            return (eigvecs * (1.0 / np.sqrt(eigvals))) @ (eigvecs.T @ z)

        # First evaluation.
        fvals = [func(np.clip(m, -5, 5))]
        calls = 1
        best_x = np.clip(m, -5, 5).copy()
        best_f = fvals[0]

        # CMA-ES main loop.
        while calls < budget and time.time() < deadline:
            # Generate population.
            pop = np.empty((lam, dim))
            fvals = np.empty(lam)
            for i in range(lam):
                x = m + sigma * sample()
                x = np.clip(x, -5, 5)
                pop[i] = x
                fvals[i] = func(x)
                calls += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x.copy()
                if time.time() >= deadline:
                    break

            # If we hit the deadline mid-generation, stop gracefully.
            if time.time() >= deadline:
                break

            # Sort by fitness.
            order = np.argsort(fvals)
            f_sorted = fvals[order]
            x_sorted = pop[order]

            # Recombination: weighted mean of the mu best.
            m_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)

            # Update evolution paths.
            Cinv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            Cinv_sqrt_m = Cinv_sqrt @ (m_new - m)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma *
                                                        (2 - c_sigma) * mu_eff) * Cinv_sqrt_m
            h_sigma = (np.linalg.norm(p_sigma) /
                       np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lam)))) \
                < (1 + 2 / (dim + 1)) * chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * Cinv_sqrt_m

            # Update covariance matrix.
            y = x_sorted[:mu] - m
            outer = np.sum(weights[:, None, None] *
                           y[:, :, None] * y[:, None, :], axis=0)
            C = (1 + c_1 * (2 - c_1) + c_mu - c_1) * C \
                + c_1 * (np.outer(p_c, p_c) - C) \
                + c_mu * (outer - C)

            # Ensure symmetry.
            C = 0.5 * (C + C.T)

            # Eigen-decompose for sampling and step-size update.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)

            # Step-size adaptation (1/5 rule on p_sigma).
            sigma *= np.exp(c_sigma / chi_n *
                            (np.linalg.norm(p_sigma) / chi_n - 1))
            # Keep sigma in a sane range.
            sigma = np.clip(sigma, 1e-4, 20.0)

            m = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
