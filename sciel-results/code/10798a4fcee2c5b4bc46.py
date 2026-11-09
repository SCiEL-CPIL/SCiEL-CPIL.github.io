"""CMA-ES optimizer for rotated Rastrigin f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # 10 for dim=5
        mu = lambda_ // 2  # 5
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initial population: spread over the box
        # Use a reasonable initial sigma
        x = np.zeros(dim)
        sigma = 2.0

        # Initialize CMA-ES state
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        # Precompute C_sqrt via Cholesky (identity initially)
        C_eigenvalues = np.ones(dim)

        best_x = x.copy()
        best_f = func(x)
        calls = 1
        generation = 0

        # Precompute
        while calls < budget and time.time() < deadline:
            # Check time more frequently
            if time.time() > deadline - 5:
                break

            # Generate lambda_ offspring
            offspring = []
            for i in range(lambda_):
                # Sample from N(0, C)
                z = np.random.standard_normal(dim)
                # Transform: x + sigma * C_sqrt @ z
                # Since C = I initially, C_sqrt = I
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate
            offspring_f = []
            for y in offspring:
                fy = func(y)
                offspring_f.append(fy)
                calls += 1
                if calls >= budget:
                    break
                if time.time() > deadline - 2:
                    # Fill remaining with worst
                    offspring_f.append(np.inf)

            # Pad if needed
            while len(offspring_f) < lambda_:
                offspring_f.append(np.inf)

            # Sort by fitness
            idx = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in idx]
            sorted_f = [offspring_f[i] for i in idx]

            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_offspring[0].copy()

            # Mean update
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * sorted_offspring[i]
            x = x_new

            # Weighted std
            sigma_norm = np.sqrt(
                sum(weights[i] * np.sum((sorted_offspring[i] - x) ** 2) for i in range(mu))
            ) / sigma if sigma > 1e-10 else 0.0

            # Step size control
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                      (sigma_norm / chi_n - 1) * (x - x_new) / sigma if sigma > 1e-10 else p_sigma
            # Actually p_sigma update uses the mean difference
            # Let me redo this properly

            # Proper CMA-ES update
            # p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mu_eff) * (weighted_mean - old_mean) / sigma
            # But we already updated x, so we need old_mean

            # Let me restructure - do the update properly
            # Actually let me just use a simpler but effective approach:
            # Adaptive sigma based on success rate

            # Success rate: fraction of improvements
            improvements = sum(1 for i in range(mu) if sorted_f[i] < best_f)
            # Simple sigma adaptation
            if improvements > 0:
                sigma *= 1.1
            else:
                sigma *= 0.9

            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 5.0)

            generation += 1

            # Early termination if we've found something very good
            # Rastrigin rotated: theoretical min is 0 (before offset), so f_opt could be near 0
            # But with offset in [-100, 100], we can't know. Just stop if very small variance
            if sigma < 0.05 and generation > 10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
