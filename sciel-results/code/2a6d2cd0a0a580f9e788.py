"""CMA-ES optimizer for BBOB f20 (Schwefel) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Standard CMA-ES parameters
        lam = max(4 + int(3 * np.log(n)), 4)  # population size
        mu = lam // 2  # number of parents
        # Weights (proportional, normalized)
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 2)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Initial state
        x_mean = np.zeros(n)  # center, since minimizer in [-4,4]^5
        sigma = 2.0
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)  # C^{-1/2}
        C_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Track best
        best_x = x_mean.copy()
        best_f = np.inf

        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation
        f_mean = evaluate(x_mean)

        # Main loop
        while calls_made < budget:
            # Generate population
            if C_sqrt_inv is None:
                C_sqrt_inv = np.linalg.inv(C_sqrt)

            # Generate candidates
            candidates = []
            for i in range(lam):
                z = np.random.randn(n)
                y = x_mean + sigma * C_sqrt @ z
                y_clipped = np.clip(y, -5, 5)
                candidates.append(y_clipped)

            # Evaluate all candidates
            fitnesses = []
            for y in candidates:
                f = evaluate(y)
                fitnesses.append(f)
                if calls_made >= budget:
                    break

            if calls_made >= budget:
                break

            # Sort by fitness
            indices = np.argsort(fitnesses)
            selected = [candidates[i] for i in indices[:mu]]
            selected_f = [fitnesses[i] for i in indices[:mu]]

            # Weighted mean of selected
            x_new_mean = sum(w[i] * selected[i] for i in range(mu))
            x_new_mean = np.asarray(x_new_mean)

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ ((x_new_mean - x_mean) / sigma)

            # h_sigma: heuristic to detect if p_sigma is consistent with p_c
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = float(norm_p_sigma / (1 - (1 - c_sigma) ** (budget - calls_made)) < (1.4 + 2 / (n + 1)) * chi_n) if (budget - calls_made) > 0 else True
            # Simpler: just check norm
            h_sigma = norm_p_sigma < (1.4 + 2 / (n + 1)) * chi_n

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma

            # Update covariance matrix C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(w[i] * np.outer(selected[i] - x_mean, selected[i] - x_mean) for i in range(mu))

            C = (1 + c_sigma * c_sigma * (c_sigma - d_sigma) * delta_h + c_1 + c_mu - c_1 - c_mu) * C \
                + c_1 * (rank_one - C) + c_mu * rank_mu

            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2.0

            # Cholesky decomposition (with fallback)
            try:
                C_sqrt = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                # Fallback: eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Step-size adaptation
            sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1))

            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x_mean = x_new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
