"""CMA-ES based black-box optimizer for Rastrigin f3."""
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
        mu_eff = 1.0 / (weights ** 2).sum()

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Initial step size
        sigma = 2.0

        # Covariance matrix
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # CMA-ES update parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        def run_cmaes():
            nonlocal calls, best_x, best_fx, x, fx, sigma, C, C_inv_sqrt, C_sqrt, p_sigma, p_c

            while calls < budget:
                # Check time
                if time.time() > deadline - 5:
                    break

                # Generate population
                X = np.zeros((lam, dim))
                F = np.zeros(lam)
                for i in range(lam):
                    if calls >= budget:
                        break
                    step = sigma * (C_sqrt @ rng.standard_normal(dim))
                    X[i] = np.clip(x + step, -5, 5)
                    F[i] = func(X[i])
                    calls += 1
                    if F[i] < best_fx:
                        best_fx = F[i]
                        best_x = X[i].copy()

                if calls >= budget:
                    break

                # Sort by fitness
                idx = np.argsort(F)
                X_sorted = X[idx]
                F_sorted = F[idx]

                # Weighted recombination
                x_old = x.copy()
                x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
                x_new = np.clip(x_new, -5, 5)

                # Update sigma (CSA)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ (x_new - x_old))
                norm_ps = np.linalg.norm(p_sigma)
                chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e6)

                # Update C (rank-one and rank-mu updates)
                h_sigma = norm_ps / np.sqrt(1 - (1 - c_sigma) ** (budget - calls + 1)) < (1 + 2 / (dim + 1)) * chi_n
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_old) / sigma

                # Rank-one update
                C += c_1 * (np.outer(p_c, p_c) / sigma ** 2 - 1) * C
                # Rank-mu update
                y = (X_sorted[:mu] - x_old) / sigma
                C += c_mu * np.sum(weights[:, None, None] * np.outer(y[:, :, None], y[:, None, :]), axis=0)

                # Update C_sqrt via eigendecomposition (every 10 iterations for efficiency)
                if (budget - calls) % 10 == 0 or calls == 1:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                    C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T

                x = x_new

        run_cmaes()
        return best_x, best_fx
# EVOLVE-BLOCK-END
