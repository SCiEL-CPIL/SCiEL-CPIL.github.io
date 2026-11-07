"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance and step-size for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim

        # Initial point: center of the box
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        if calls_made >= self.budget:
            return best_x, best_fx

        # CMA-ES parameters
        sigma = 1.0
        C = np.eye(n)
        
        # Population size
        lambda_pop = 4 + int(np.floor(3 * np.log(n)))
        # Learning rates
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        gen = 0

        while calls_made < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Check if we have enough budget for a full generation
            current_lambda = min(lambda_pop, self.budget - calls_made)
            if current_lambda < 2:
                # Not enough points for a meaningful update, just evaluate and break
                # Actually, if we have 1 call left, we can evaluate it, update best, and break.
                # But we can't update CMA-ES.
                # Let's just evaluate it if we have exactly 1 call left.
                if current_lambda == 1:
                    y = rng.multivariate_normal(np.zeros(n), C, size=1).flatten()
                    x_try = np.clip(x + sigma * y, -5, 5)
                    f_try = func(x_try)
                    calls_made += 1
                    if f_try < best_fx:
                        best_fx = f_try
                        best_x = x_try.copy()
                break

            # Sample population
            Y = rng.multivariate_normal(np.zeros(n), C, size=current_lambda)

            # Evaluate population
            Fs = []
            for i in range(current_lambda):
                x_try = np.clip(x + sigma * Y[i], -5, 5)
                f_try = func(x_try)
                Fs.append(f_try)
                calls_made += 1
                if f_try < best_fx:
                    best_fx = f_try
                    best_x = x_try.copy()

                if calls_made >= self.budget:
                    break

            Fs = np.array(Fs)
            
            # If we didn't complete the population, skip update
            if len(Fs) < current_lambda:
                break

            # Sort
            order = np.argsort(Fs)
            Y_sorted = Y[order]

            # Update mean
            mean_new = x + sigma * sum(weights[i] * Y_sorted[i] for i in range(mu))
            mean_new = np.clip(mean_new, -5, 5)

            # Update covariance
            C_new = (1 - c_1 - c_mu) * C
            for i in range(mu):
                C_new += weights[i] * np.outer(Y_sorted[i], Y_sorted[i])

            # Update step size
            delta_mean = (mean_new - x) / sigma
            p_sigma_new = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_mean
            p_sigma_norm = np.linalg.norm(p_sigma_new)
            
            # hsig condition
            # Using gen count for the expected norm
            hsig = int(p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / chi_n < (1.4 + 2 / (n + 1)) * chi_n)

            if hsig:
                p_c_new = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_mean
                C_new += (c_c * (2 - c_c) * mu_eff - c_1 - c_mu) * np.outer(p_c_new, p_c_new)
            else:
                p_c_new = (1 - c_c) * p_c

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update state
            C = C_new
            p_sigma = p_sigma_new
            p_c = p_c_new
            x = mean_new
            gen += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
