"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with box-constraint handling for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff)
        c_mu = min(1 - c_1,
                   2 * (mu_eff - 2 + 1 / (dim + 2)) / ((dim + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        best_x = x_mean.copy()
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial evaluation
        fx0 = evaluate(x_mean)

        while calls < budget:
            if time.time() > deadline - 5.0:
                break

            # Generate population
            population = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                population[i] = x_mean + sigma * (B @ (D * z))
                population[i] = np.clip(population[i], -5, 5)

            # Evaluate
            fitness = np.array([evaluate(population[i]) for i in range(lam)])

            # Sort by fitness
            idx = np.argsort(fitness)
            y = population[idx]
            fitness_sorted = fitness[idx]

            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * y[:mu], axis=0)

            # Update covariance matrix adaptation
            y_inv = np.zeros((mu, dim))
            for i in range(mu):
                y_inv[i] = (y[i] - x_mean_old) / (sigma * np.sqrt(np.dot(B.T, (B @ (D ** 2 * (B.T @ ((y[i] - x_mean_old) / sigma)))))))
                # Simpler: compute using the inverse transform
            # Actually let's use a simpler approach for the covariance update
            # using the standard CMA-ES update

            # Recompute y_inv properly
            for i in range(mu):
                diff = (y[i] - x_mean_old) / sigma
                # Inverse transform: C^{-1/2} diff
                y_inv[i] = B.T @ (D ** (-1) * (B.T @ diff))

            delta_sig = np.sum(weights * y_inv, axis=0)
            hsig = (np.linalg.norm(delta_sig) /
                    np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / (dim + c_sigma)))) < \
                   (1 + 2 / (dim + 1)) * chi_n

            ps = (1 - c_sigma) * ps + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_sig

            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean_old) / sigma

            # Update C
            C = ((1 + c_1 * (2 - c_1) - c_mu) * C -
                 c_1 * np.outer(pc, pc) +
                 c_mu * np.sum(weights[:, None, None] * np.outer(y_inv, y_inv), axis=0))

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Eigen-decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                B = np.eye(dim)
                D = np.ones(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n *
                            (np.linalg.norm(ps) / chi_n - 1))

            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 1e6)

            # Stopping criteria
            if sigma < 1e-12:
                break

            # Check for convergence
            if np.max(fitness_sorted) - np.min(fitness_sorted) < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
