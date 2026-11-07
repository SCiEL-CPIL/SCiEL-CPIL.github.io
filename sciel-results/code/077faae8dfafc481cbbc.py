"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance for bent-cigar (f12) optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initial point and evaluation
        x = np.zeros(dim)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        call_count = 1

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # ~10 for dim=5
        # Step size
        sigma = 2.5
        # Mean
        mean = x.copy()

        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Cholesky decomposition for sampling
        L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Learning rates
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Counter for stagnation detection
        no_improve_count = 0
        stagnation_threshold = 30

        while call_count < budget:
            # Check time
            if time.time() > deadline:
                break

            # Sample offspring
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = mean + sigma * L @ z
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                if call_count >= budget:
                    break
                fy = func(y)
                call_count += 1
                offspring_f.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            if len(offspring_f) < lam:
                # Pad with last value if we ran out of budget
                while len(offspring_f) < lam:
                    offspring_f.append(offspring_f[-1] if offspring_f else 0.0)
                    offspring.append(offspring[-1] if offspring else mean)

            # Sort by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]

            # Update mean
            new_mean = sum(weights[i] * sorted_offspring[i] for i in range(mu))

            # Update sigma
            z_mean = (new_mean - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff) else 0.0
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * z_mean
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_c * (2 - c_c) - c_1 * delta_h) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            C += c_mu * sum(weights[i] * np.outer(sorted_offspring[i] - mean, sorted_offspring[i] - mean) for i in range(mu))

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2

            # Update Cholesky
            try:
                L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            mean = new_mean

            # Stagnation restart
            if no_improve_count > stagnation_threshold:
                # Local restart: perturb mean
                mean = np.clip(mean + 0.5 * rng.standard_normal(dim), -5, 5)
                sigma = max(sigma, 0.5)
                C = np.eye(dim)
                L = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0
                f_mean = func(mean)
                call_count += 1
                if f_mean < best_f:
                    best_f = f_mean
                    best_x = mean.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
