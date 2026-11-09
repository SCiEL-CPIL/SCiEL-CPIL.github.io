"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f16 (Weierstrass) with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng()

        # CMA-ES parameters: lambda=10 gives ~50 generations with 500 budget
        # Slightly larger population for better covariance estimation in 5D
        lambda_ = 10
        mu = lambda_ // 2  # top 5 individuals

        # Logarithmic weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Standard CMA-ES adaptation rates
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + (4 + dim / (dim + 4)))
        cs = (dim + 1.7) / (dim + 9)
        c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - cc)
        cmu = min(1 - c1, (2 * (mu_eff - 2 + 1 / mu_eff)) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        # Initialize at center with moderate initial sigma
        # Weierstrass has global minimum shifted; start broad to find basin
        x_mean = np.zeros(dim)
        sigma = 3.0  # Wide enough to explore [-5,5]^5 initially
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        evals_used = 0
        generation = 0
        best_x = x_mean.copy()
        best_f = float('inf')

        def evaluate(x):
            nonlocal evals_used, best_x, best_f
            fx = func(x)
            evals_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial evaluation at origin
        fx0 = evaluate(x_mean)

        # Main CMA-ES loop
        while evals_used + lambda_ <= budget:
            generation += 1

            # Generate population
            x_pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                x_pop[i] = x_mean + sigma * (B @ (D * z))

            # Evaluate all offspring
            f_pop = np.array([evaluate(x_pop[i]) for i in range(lambda_)])

            # Sort by fitness (ascending)
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]

            # Weighted recombination of top mu
            x_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)

            # Update path evolution strategies
            # ps in transformed space
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (x_new - x_mean)))
            ps_norm = np.linalg.norm(ps)

            # hsig: check if ps is not too large (no drift)
            hsig = ps_norm / np.sqrt(1 - (1 - cs) ** (2 * generation)) < (1.4 + 2 / (dim + 1)) * chi_n

            # Update pc (mean path)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new - x_mean)

            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, None] * np.outer(x_sorted[:mu] - x_mean, x_sorted[:mu] - x_mean), axis=0)

            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

            # Symmetrize and ensure positive semi-definite
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-18, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            B = eigvecs
            D = np.sqrt(eigvals)

            # Update sigma using cs and damping
            sigma *= np.exp(cs / damps * (ps_norm / chi_n - 1))

            # Update mean
            x_mean = x_new

            # Early stopping if sigma is very small
            if sigma < 1e-14:
                break

        # Use remaining budget for intensive local refinement
        remaining = budget - evals_used
        if remaining > 0:
            # Start with small step relative to current sigma
            local_sigma = max(sigma * 0.3, 0.05)
            for _ in range(remaining):
                z = rng.standard_normal(dim)
                x_trial = best_x + local_sigma * z
                f_trial = evaluate(x_trial)
                if f_trial < best_f:
                    # Success: keep step size or slightly reduce
                    local_sigma *= 0.97
                else:
                    # Failure: reduce step size more aggressively
                    local_sigma *= 0.93

        return best_x, best_f
# EVOLVE-BLOCK-END
