"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start CMA-ES-like optimizer with a (mu+lambda) strategy.
    Uses adaptive step-size control, a time guard, and restarts.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s

        best_x = None
        best_f = np.inf

        # We will run multiple restarts of a population-based optimizer.
        # Each restart gets a portion of the budget.
        # Let's use a strategy where we try a few different initializations.

        # Parameters for the (mu+lambda) CMA-ES
        # lambda = number of offspring
        # mu = number of best offspring used for update
        lam = 12
        mu = lam // 2
        weights = np.ones(mu)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Adaptation rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 2) / (dim + 2) - 1))

        # Step size
        sigma0 = 1.0

        # Restart budget allocation
        # We'll try to do several restarts.
        # Let's say we do 3 restarts, each with budget/3.
        # But we need to be careful with the time limit.
        n_restarts = 3
        budget_per_restart = total_budget // n_restarts

        for restart_idx in range(n_restarts):
            if time.time() > deadline:
                break
            if total_budget <= 0:
                break

            # Initialize population center
            # For the first restart, try origin and a random point.
            # For subsequent restarts, random points.
            if restart_idx == 0:
                x = np.zeros(dim)
            elif restart_idx == 1:
                rng = np.random.default_rng(1234 + restart_idx)
                x = rng.uniform(-4, 4, dim)
            else:
                rng = np.random.default_rng(1234 + restart_idx)
                x = rng.uniform(-4, 4, dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                total_budget -= 1

            if total_budget <= 0 or time.time() > deadline:
                break

            # CMA-ES state
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)

            # We need to manage the budget carefully.
            # Each generation costs lam evaluations.
            # We have budget_per_restart - 1 evaluations left for this restart.
            # But we also need to account for the time limit.

            gen = 0
            while total_budget >= lam and time.time() < deadline:
                gen += 1
                # Generate offspring
                Y = np.empty((lam, dim))
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    Y[i] = x + sigma * (B @ (D * z))
                    # Clip to box
                    Y[i] = np.clip(Y[i], -5, 5)

                # Evaluate offspring
                F = np.empty(lam)
                for i in range(lam):
                    F[i] = func(Y[i])
                    total_budget -= 1

                # Update best
                for i in range(lam):
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = Y[i].copy()

                if time.time() > deadline:
                    break

                # Sort offspring by fitness
                order = np.argsort(F)
                Y_sorted = Y[order]
                F_sorted = F[order]

                # Update mean
                y_w = np.zeros((mu, dim))
                for i in range(mu):
                    y_w[i] = Y_sorted[i] - x
                y_w_sigma = y_w / sigma
                x_new = x + sigma * (weights @ y_w_sigma)
                # Clip mean to box? CMA-ES usually doesn't clip the mean,
                # but the offspring are clipped. Let's clip the mean to be safe.
                x_new = np.clip(x_new, -5, 5)

                # Update step size (CSA)
                h_sigma = (
                    np.sqrt((1 - np.exp(-2 / (c_sigma * (gen + 1)))) *
                            (1 - np.exp(-4 / (c_sigma * (gen + 1))))) *
                    np.sqrt(dim - 1) / np.sqrt(2 * dim)
                )
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma)) * (B.T @ (D * (weights @ y_w_sigma)))
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
                # Simpler step size update:
                # sigma *= np.exp(0.3 * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))

                # Update covariance matrix
                h_c = (
                    np.sqrt((1 - np.exp(-2 / (c_c * (gen + 1)))) *
                            (1 - np.exp(-6 / (c_c * (gen + 1))))) *
                    np.sqrt(dim + 1) / np.sqrt(2)
                )
                p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c)) * (B.T @ (D * (weights @ y_w_sigma)))

                # Rank-one update
                y_w_sigma_mean = np.zeros(dim)
                for i in range(mu):
                    y_w_sigma_mean += weights[i] * (y_w_sigma[i] - x)  # This is not quite right for rank-one
                # Actually, rank-one update uses the weighted mean of the selected y_w_sigma
                y_w_sigma_mean = weights @ y_w_sigma
                C = (1 + c_1 * c_mu - c_1) * C + c_1 * (y_w_sigma_mean[:, None] @ y_w_sigma_mean[None, :])

                # Rank-mu update
                C = (1 + c_mu * c_mu - c_mu) * C + c_mu * (y_w_sigma @ y_w_sigma.T)

                # Ensure positive semi-definite
                # We'll use a simple check and regularization if needed
                # For simplicity, we'll just proceed and hope it's okay.
                # A more robust approach would be to do a Cholesky decomposition and fix if it fails.

                # Update B and D
                # We need to compute the Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
                    B = L
                    D = 1.0 / np.diag(L)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, reset C to identity and recompute
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)

                x = x_new

                # Check for convergence
                if np.linalg.norm(sigma * D) < 1e-10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
