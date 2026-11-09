"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step sizes and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        remaining = self.budget

        # Number of restarts: we'll do several restarts with decreasing budgets
        # For dim=5, a single run might be enough, but restarts help with bi-modal landscapes
        n_restarts = 3
        # Allocate budget: first restart gets more, subsequent get less
        # But we need to be careful not to exceed total budget
        # Let's do: restart 1 gets 60%, restart 2 gets 25%, restart 3 gets 15%
        budgets = []
        for i in range(n_restarts):
            if i == 0:
                b = int(remaining * 0.55)
            elif i == 1:
                b = int(remaining * 0.30)
            else:
                b = remaining
            budgets.append(b)
            remaining -= b

        # Ensure we don't go over
        total_alloc = sum(budgets)
        if total_alloc > self.budget:
            # Scale down
            budgets = [int(b * self.budget / total_alloc) for b in budgets]

        for restart_idx, rbudget in enumerate(budgets):
            if rbudget < 10:
                break
            x, f, calls_used = self._cmaes_restart(func, rng, rbudget, best_x, best_f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            remaining -= calls_used
            if remaining <= 0:
                break

        if best_x is None:
            # Fallback: just do a few random evaluations
            x = rng.uniform(-5, 5, self.dim)
            best_f = func(x)
            best_x = x

        return best_x, best_f

    def _cmaes_restart(self, func, rng, budget, init_x, init_f):
        dim = self.dim
        if init_x is not None and init_f < np.inf:
            x = init_x.copy()
            fx = init_f
            calls_used = 0
        else:
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_used = 1

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # ~13 for dim=5
        mu = lambda_ // 2  # number of best solutions used for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size
        sigma = 2.0
        # Initial covariance matrix
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Update rules
        c_sigma = (mu_eff + 2) / (dim + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] + weights[1] - 1) / ((dim + 2) ** 2 + mu_eff))

        # Chi distribution approximation
        chi = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        best_x = x.copy()
        best_f = fx

        for _ in range(budget - calls_used):
            # Generate population
            ys = []
            fs = []
            for i in range(lambda_):
                # Sample from N(x, sigma^2 * C)
                # Use Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: use identity if C is not positive definite
                    L = np.eye(dim)
                    C = np.eye(dim)

                z = rng.standard_normal(dim)
                y = x + sigma * L @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                ys.append(y)
                fs.append(fy)

            fs = np.array(fs)
            ys = np.array(ys)

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()

            # Recombination
            y_w = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)

            # Update evolution paths
            y_diff = (y_w - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_diff
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2)) / chi < (1.4 + 2 / (dim + 1)) * 1.0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_diff

            # Update covariance matrix
            delta_h = (1.0 if h_sigma else 0.0) * (1 - c_c) * c_c
            C = ((1 + c_1 + c_mu - c_1 * dim) * C +
                 c_1 * (np.outer(p_c, p_c) - C) +
                 delta_h * C +
                 c_mu * np.sum(weights[:, None, None] * np.outer(ys_sorted[:mu] - x, ys_sorted[:mu] - x) / sigma ** 2, axis=0))

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = y_w

        return best_x, best_f, calls_used
# EVOLVE-BLOCK-END
