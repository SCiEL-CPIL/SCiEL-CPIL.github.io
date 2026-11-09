"""Rewritten program using multi-start CMA-ES for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start CMA-ES with restarts for robust global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        remaining = self.budget

        # Multi-start: run several short CMA-ES instances
        # First start gets more budget, subsequent ones get less
        n_starts = 3
        # Budget allocation: e.g., 60%, 25%, 15% of remaining
        allocs = [0.6, 0.25, 0.15]
        # Adjust for small budgets
        if self.budget < 100:
            n_starts = 1
            allocs = [1.0]
        elif self.budget < 200:
            n_starts = 2
            allocs = [0.7, 0.3]

        for s in range(n_starts):
            if remaining <= 0:
                break
            # Determine budget for this start
            if s == n_starts - 1:
                budget_s = remaining
            else:
                budget_s = max(10, int(allocs[s] * self.budget))
                budget_s = min(budget_s, remaining)

            # Initialize CMA-ES
            if s == 0:
                # First start: random center
                x0 = rng.uniform(-5, 5, self.dim)
                sigma0 = 2.0
            else:
                # Subsequent starts: center near best found, with perturbation
                if best_x is not None:
                    x0 = np.clip(best_x + rng.normal(0, 1.0, self.dim), -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, self.dim)
                sigma0 = 1.0

            # Run CMA-ES for budget_s evaluations
            x_c, f_c, calls_used = self._cmaes(func, x0, sigma0, budget_s, rng)

            if f_c < best_f:
                best_f = f_c
                best_x = x_c.copy()

            remaining -= calls_used

        return best_x, best_f

    def _cmaes(self, func, x0, sigma, budget, rng):
        """Run CMA-ES with the given initial point and step size."""
        n = self.dim
        calls = 0
        lambda_ = 4 + int(3 * np.log(n))  # population size
        lambda_ = min(lambda_, budget // 2) if budget > 10 else 2
        mu = lambda_ // 2  # number of parents
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # State
        x_mean = np.clip(x0.copy(), -5, 5)
        C = np.eye(n)
        C_inv = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        f_mean = func(x_mean)
        calls += 1
        best_x = x_mean.copy()
        best_f = f_mean

        # Track best for early stopping
        no_improve = 0
        max_no_improve = max(10, budget // 10)

        while calls < budget:
            # Generate offspring
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if C is not positive definite
                C = np.eye(n)
                L = np.eye(n)

            # Sample
            z = rng.standard_normal((lambda_, n))
            y = z @ L.T  # y ~ N(0, C)
            y_sigma = sigma * y
            # Clip to box
            x_offspring = np.clip(x_mean + y_sigma, -5, 5)

            # Evaluate
            f_offspring = np.empty(lambda_)
            for i in range(lambda_):
                f_offspring[i] = func(x_offspring[i])
                calls += 1
                if calls >= budget:
                    break

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(f_offspring)
            y_sorted = y[order]
            f_sorted = f_offspring[order]
            x_sorted = x_offspring[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

            # Update mean
            x_mean_new = x_mean + sigma * np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            x_mean = np.clip(x_mean_new, -5, 5)

            # Update evolution paths
            # Check if C is still PD
            try:
                C_inv_sqrt = np.linalg.cholesky(C_inv)
                y_tilde = (C_inv_sqrt.T @ (x_mean_new - x_mean) / sigma).T
            except np.linalg.LinAlgError:
                y_tilde = (x_mean_new - x_mean) / sigma

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_tilde
            norm_ps = np.linalg.norm(ps)
            h_sig = norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lambda_ + 1))) < (1 + 2 / (n + 1)) * chi_n

            pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_tilde

            # Update covariance matrix
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            rank_one = np.outer(ps, ps)
            rank_mu = np.sum(weights[:, None, None] * y_sorted[:mu].T @ y_sorted[:mu].T, axis=0)
            C = (1 + c_sigma * c_sigma * (2 - c_sigma) * delta_h - c_1 - c_mu) * C \
                + c_1 * (rank_one + delta_h * np.outer(pc, pc)) + c_mu * rank_mu

            # Update inverse
            C_inv = np.linalg.inv(C)

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

            # Ensure sigma is reasonable
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Check for convergence
            if sigma < 1e-10:
                break

        return best_x, best_f, calls
# EVOLVE-BLOCK-END
