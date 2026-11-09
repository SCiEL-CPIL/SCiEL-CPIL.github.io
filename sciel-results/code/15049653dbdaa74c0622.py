"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multiple restarts for Rastrigin-style multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # CMA-ES implementation
        def cma_es(x0, sigma0, budget_cma, deadline, rng, func, best_x_ref, best_f_ref):
            """Run CMA-ES from x0 with initial sigma sigma0 for budget_cma evaluations."""
            n = len(x0)
            # Population size
            lambda_ = 4 + int(3 * np.log(n))
            mu = lambda_ // 2

            # Weights for recombination
            w = np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1 / np.sum(w ** 2)

            # Strategy parameters
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

            # State variables
            xmean = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            invsqrtC = np.eye(n)
            # Cholesky factor of C
            D = np.linalg.cholesky(C)
            B = np.linalg.inv(D)
            # Precompute for efficiency
            Cinv_sqrt = np.linalg.inv(D.T @ D)

            evals = 0
            best_x_local = xmean.copy()
            best_f_local = np.inf

            while evals < budget_cma:
                if time.time() > deadline:
                    break

                # Generate population
                population = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    y = xmean + sigma * (D @ z)
                    y = np.clip(y, -5, 5)
                    population.append(y)
                    evals += 1

                # Evaluate
                fitness = np.array([func(x) for x in population])

                # Update best
                idx_best = np.argmin(fitness)
                if fitness[idx_best] < best_f_local:
                    best_f_local = fitness[idx_best]
                    best_x_local = population[idx_best].copy()
                if best_f_local < best_f_ref[0]:
                    best_f_ref[0] = best_f_local
                    best_x_ref[0] = best_x_local.copy()

                # Sort by fitness
                order = np.argsort(fitness)
                y_sorted = [population[i] for i in order]
                f_sorted = fitness[order]

                # Update mean
                y_mean = np.zeros(n)
                for i in range(mu):
                    y_mean += w[i] * y_sorted[i]
                xmean_new = xmean + sigma * y_mean
                xmean_new = np.clip(xmean_new, -5, 5)

                # Update covariance matrix
                # Compute C update
                yv = np.array([y_sorted[i] - y_mean for i in range(mu)])
                delta_x = (xmean_new - xmean) / sigma if sigma > 1e-15 else np.zeros(n)

                # Update path covariance
                pc = (1 - cc) * pc + cc * delta_x

                # Update path step
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC.T @ delta_x

                # Hessian correction
                chi_n = np.sqrt(n) * (1 + 1 / (2 * n) + 1 / (8 * n ** 2))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (evals + 1)) / chi_n < (1.4 + 2 / (n + 1)) and 1)

                # Update C
                C = (1 + c1 * hsig * (cc - 1) - c1 - cmu) * C \
                    + c1 * (pc @ pc.T + (hsig and (cc - 1) * cc) or 0) \
                    + cmu * np.sum([w[i] * (yv[i] @ yv[i].T) for i in range(mu)], axis=0)

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                # Regularize
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update Cholesky
                D = np.linalg.cholesky(C)
                B = np.linalg.inv(D)
                invsqrtC = B

                xmean = xmean_new

                # Stagnation check: if no improvement in last lambda_ evaluations
                if evals > lambda_ * 3:
                    recent_best = np.min(f_sorted[:min(lambda_, len(f_sorted))])
                    if abs(recent_best - best_f_local) < 1e-10:
                        break

            return best_x_local, best_f_local, evals

        # Multi-restart CMA-ES
        num_restarts = 4
        # First restart gets more budget for exploration
        budgets = [150, 120, 100, 80]  # Total: 450, leaving 50 for final refinement
        init_points = [
            np.zeros(dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        for restart_idx in range(num_restarts):
            if time.time() > deadline:
                break

            x0 = init_points[restart_idx].copy()
            sigma0 = 1.0 if restart_idx == 0 else 0.5

            bx, bf, used = cma_es(x0, sigma0, budgets[restart_idx], deadline, rng, func,
                                   [best_x, best_f] if best_x is not None else [None, np.inf],
                                   [best_f] if best_f < np.inf else [np.inf])

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

            # Update budget tracking
            budget -= used

            if time.time() > deadline:
                break

        # Final local refinement around the best point
        remaining = budget
        if remaining > 10 and best_x is not None:
            # Simple (1+1)-ES refinement
            x = best_x.copy()
            fx = best_f
            sigma = 0.3

            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.7

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                sigma = np.clip(sigma, 0.001, 1.0)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
