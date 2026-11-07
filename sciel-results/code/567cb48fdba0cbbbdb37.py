# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize at center
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1

        # CMA-ES parameters
        sigma = 2.0
        lambda_ = 4 * dim + 4  # population size
        # Covariance matrix C = I initially
        C = np.eye(dim)
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Adaptation parameters
        cs = (lambda_ + 2) / (dim + lambda_ + 5)
        ds = 1 + 2 * max(0, (np.sqrt((lambda_ + 1) / (dim + 1)) - 1) / (dim + 2)) + cs
        cmu = min(0.1 / (dim + 1), 0.1)
        cc = (4 + dim / lambda_) / (dim + 4 + 2 * lambda_ / dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Cholesky factor of C
        C_chol = np.linalg.cholesky(C)

        # Track progress for step size adaptation
        progress_window = []
        W = lambda_ // 2  # weights: top half get weight 1, rest get 0 (simple)
        mu_eff = (sum(np.ones(W)) ** 2) / sum(np.ones(W) ** 2)  # = W for uniform
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (np.sqrt((mu_eff + 1) / (dim + 1)) - 1) / (dim + 2)) + c_sigma

        for iteration in range(budget - 1):
            # Time check
            if time.time() > deadline:
                break

            # Generate lambda_ offspring
            Y = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x + sigma * C_chol @ z
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                call_count += 1

                # Early time check inside loop
                if time.time() > deadline:
                    break

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update best
            if F_sorted[0] < best_fx:
                best_fx = F_sorted[0]
                best_x = Y_sorted[0].copy()

            # Check if we should stop (convergence or time)
            if time.time() > deadline:
                break

            # Weighted recombination
            w = np.zeros(lambda_)
            w[:W] = 1.0 / W
            mu_eff = (w[:W] ** 2).sum() ** (-1) * (w[:W].sum() ** 2)

            # Update evolution paths
            z_mean = np.zeros(dim)
            for i in range(W):
                z_i = np.linalg.solve(C_chol, (Y_sorted[i] - x) / sigma)
                z_mean += w[i] * z_i

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (iteration + 1)))) / chiN < (1.4 + 2 / (dim + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x - sigma * C_chol @ z_mean)
            x = x + sigma * (cc + 1) * pc + sigma * np.linalg.solve(C_chol, z_mean)

            # Update covariance
            delta_d = (hsig and (1 + 2 * (cc - 1))) or 1
            rank1 = np.outer(ps, ps)
            rankmu = np.zeros((dim, dim))
            for i in range(W):
                z_i = np.linalg.solve(C_chol, (Y_sorted[i] - x) / sigma)
                rankmu += w[i] * np.outer(z_i, z_i)
            C = (1 + cmu * (mu_eff - 2 + delta_d) - cs * (2 - cs)) * C + cmu * (rankmu + delta_d * rank1)

            # Ensure C is positive definite
            C = (C + C.T) / 2
            try:
                C_chol = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_chol = np.eye(dim)

            # Adapt sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-8, 50)

            # Clip x to bounds
            x = np.clip(x, -5, 5)

            # If we're stuck, do a random restart
            if iteration % 100 == 99 and best_fx > 0:
                # Try a few random points
                for _ in range(3):
                    xr = rng.uniform(-5, 5, dim)
                    fxr = func(xr)
                    call_count += 1
                    if fxr < best_fx:
                        best_fx = fxr
                        best_x = xr.copy()
                        x = xr.copy()
                        sigma = 2.0
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
