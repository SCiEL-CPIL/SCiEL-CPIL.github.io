import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Time guard: keep a safety margin before the 300s wall-clock limit.
        start_time = time.time()
        deadline = start_time + 270.0

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))
        sigma = 2.0
        x_mean = np.zeros(n)
        C = np.eye(n)

        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        best_x = x_mean.copy()
        best_f = float('inf')

        evals_left = budget
        E_norm = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Track best improvement for restart logic
        best_f_prev = best_f
        gen = 0
        stagnation_count = 0

        while evals_left > 0 and time.time() < deadline:
            # If we've been stuck for many generations, try a restart
            if stagnation_count > 20 and evals_left > lambda_ * 5:
                # Restart: reset C, p_sigma, p_c, and increase sigma
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                sigma = max(sigma * 2.0, 1.0)
                stagnation_count = 0
                # Keep x_mean at best_x to continue from best point
                x_mean = best_x.copy()

            # Compute Cholesky factor of C
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)

            # Sample lambda points
            z = np.random.randn(lambda_, n)
            y = sigma * (z @ B.T)
            x = x_mean[None, :] + y

            # Check budget
            if evals_left < lambda_:
                for i in range(evals_left):
                    f = func(x[i])
                    if f < best_f:
                        best_f = f
                        best_x = x[i].copy()
                    evals_left -= 1
                break

            f = np.array([func(x[i]) for i in range(lambda_)])
            evals_left -= lambda_
            gen += 1

            # Sort by fitness
            idx = np.argsort(f)
            x_sorted = x[idx]
            f_sorted = f[idx]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()

            # Check for stagnation
            if best_f < best_f_prev - 1e-10:
                stagnation_count = 0
            else:
                stagnation_count += 1
            best_f_prev = best_f

            # Recombination
            x_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)

            # Compute C^{-1/2} using eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            d_sigma_vec = C_inv_sqrt @ ((x_new - x_mean) / sigma)

            # h_sigma: standard heuristic
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * gen)) < (1 + 2 / (n + 1)) * E_norm else 0.0

            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_sigma_vec

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * ((x_new - x_mean) / sigma)

            # Update C
            y_sorted = (x_sorted[:mu] - x_mean[None, :]) / sigma
            C_mu = np.sum(weights[:, None, None] * y_sorted[:, :, None] * y_sorted[:, None, :], axis=0)

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * C_mu

            # Update mean
            x_mean = x_new

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / E_norm - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals_check = np.linalg.eigvalsh(C)
            if np.any(eigvals_check < 1e-12):
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                sigma = 2.0

        return best_x, best_f
