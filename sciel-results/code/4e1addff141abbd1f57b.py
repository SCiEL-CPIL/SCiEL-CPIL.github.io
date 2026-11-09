"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f12 (bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, keep 20s safety margin
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))  # ~11 for dim=5
        mu = lambda_pop // 2  # number of parents
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # Step size control
        cs = (mueff + 1.5) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        # Covariance matrix adaptation
        cmac = (2 + 2 * mueff / (dim + 1)) / ((dim + 2) ** 2 + mueff)
        # Step size
        sigma0 = 1.0
        # Initial point - try center first since minimizer is in [-4,4]^5
        x_mean = np.zeros(dim)
        # Evaluate initial point
        fvals = []
        fvals.append(func(x_mean))
        calls = 1
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim ** 2))

        # Keep track of best
        x_best = x_mean.copy()
        f_best = fvals[0]

        # Main loop
        while calls < budget:
            # Check deadline
            if time.time() > deadline - 2.0:
                break

            # Generate population
            pop = np.zeros((lambda_pop, dim))
            for i in range(lambda_pop):
                # Sample from multivariate normal with covariance C and step size sigma
                # Use Cholesky decomposition for efficiency
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes singular
                    C = np.eye(dim)
                    L = np.eye(dim)
                z = np.random.randn(dim)
                pop[i] = x_mean + sigma * L @ z

            # Evaluate population
            pop_fvals = []
            for i in range(lambda_pop):
                if calls >= budget:
                    break
                if time.time() > deadline - 1.0:
                    break
                fi = func(pop[i])
                pop_fvals.append(fi)
                calls += 1
                if fi < f_best:
                    f_best = fi
                    x_best = pop[i].copy()

            if len(pop_fvals) < lambda_pop:
                # Not enough evaluations, break
                break

            # Sort by fitness
            idx = np.argsort(pop_fvals)
            pop = pop[idx]
            pop_fvals = np.array(pop_fvals[idx])

            # Recombination
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * pop[i]

            # Update evolution paths
            C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C))
            pc = (1 - cs) * pc + cs * C_sqrt_inv @ (x_new - x_mean) / sigma
            norm_pc = np.linalg.norm(pc)
            hsig = (norm_pc / np.sqrt(1 - (1 - cs) ** (2 * (calls - lambda_pop + 1) / (dim * cs))) <
                    (1.4 + 2 / (dim + 1)) * chi_n)

            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * cs * (1 - cs)) * C_sqrt_inv @ (x_new - x_mean) / sigma
            norm_ps = np.linalg.norm(ps)

            # Update covariance matrix
            delta_hsig = (1 - hsig) * (1 + (1 + cmac) * ds)
            rank_one = np.outer(ps, ps)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = C_sqrt_inv @ (pop[i] - x_mean) / sigma
                rank_mu += w[i] * np.outer(y_i, y_i)

            C = (1 + cmac * (mueff - 1) + cmac * delta_hsig - cmac) * C + \
                cmac * (rank_mu + delta_hsig * rank_one)

            # Update step size
            sigma *= np.exp(cs / ds * (norm_ps / chi_n - 1))
            sigma = max(1e-12, min(sigma, 1e6))

            # Update mean
            x_mean = x_new

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < 1e-10):
                C += (1e-10 - np.min(eigvals)) * np.eye(dim)

        return x_best, f_best
# EVOLVE-BLOCK-END
