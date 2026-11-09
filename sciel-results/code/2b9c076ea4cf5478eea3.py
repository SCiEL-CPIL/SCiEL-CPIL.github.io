"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 8  # population size
        mu = lam // 2  # number of selected parents
        weights = np.ones(mu)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        def cma_es(x0, sigma0, remaining_budget, local_seed=0):
            x = x0.copy()
            sigma = sigma0
            ps = np.zeros(n)
            pc = np.zeros(n)
            C = np.eye(n)

            # Pre-compute initial C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            best_x = x0.copy()
            best_f = func(x0)
            evals = 1
            local_rng = np.random.default_rng(42 + local_seed)

            for gen in range(remaining_budget // lam + 1):
                if time.time() > deadline:
                    break
                if evals >= remaining_budget:
                    break

                # Periodic eigendecomposition (every 5 generations) to save time
                if gen % 5 == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                # Sample lambda points
                Y = local_rng.standard_normal((lam, n))
                X = x[None, :] + sigma * (C_inv_sqrt @ Y.T).T
                X = np.clip(X, -5.0, 5.0)

                # Evaluate
                Fs = np.zeros(lam)
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    if evals >= remaining_budget:
                        break
                    Fs[i] = func(X[i])
                    evals += 1
                    if Fs[i] < best_f:
                        best_f = Fs[i]
                        best_x = X[i].copy()

                # Sort by fitness
                order = np.argsort(Fs)
                X_sorted = X[order]

                # Update mean
                x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

                # Update sigma (CSA)
                diff = (x_new - x) / (sigma * chi_n)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff

                # Update pc (CM)
                hsig = 1.0
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

                # Rank-mu update of C
                v = (X_sorted[:mu] - x[None, :]) / sigma
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(ps, ps) + (1 - hsig) * (1 - c_c) * c_c * C)
                C += c_mu * np.sum(weights[:, None, None] * v[:, :, None] * v[:, None, :], axis=0)
                C = 0.5 * (C + C.T)

                # Adapt sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                x = x_new
                x = np.clip(x, -5.0, 5.0)

            return best_x, best_f, evals

        # Multi-restart strategy: 4 runs for better multimodal coverage
        # Budget split: 45%, 25%, 18%, 12%
        splits = [
            int(budget * 0.45),
            int(budget * 0.25),
            int(budget * 0.18),
            budget - int(budget * 0.45) - int(budget * 0.25) - int(budget * 0.18)
        ]
        splits = [max(s, 10) for s in splits]

        best_x = None
        best_f = float('inf')

        for restart in range(4):
            if time.time() > deadline:
                break
            if restart == 0:
                x0 = rng.uniform(-4, 4, n)
                sigma0 = 2.5
            elif restart == 1:
                x0 = rng.uniform(-4, 4, n)
                sigma0 = 1.5
            else:
                # Restart from best found so far with perturbation
                x0 = best_x + 0.4 * rng.standard_normal(n)
                x0 = np.clip(x0, -4.5, 4.5)
                sigma0 = 0.5

            rx, rf, used = cma_es(x0, sigma0, splits[restart], local_seed=restart * 100)
            if rf < best_f:
                best_f = rf
                best_x = rx

            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
