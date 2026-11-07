"""CMA-ES optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (isotropic initialization) with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before the 300 s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # ---- CMA-ES parameters (isotropic initialization) ----
        # Population size ~4*dim (CMA-ES default scaling for dim=5 -> ~20)
        lam = max(4, 4 * dim)
        # Adaptation rates (standard CMA-ES defaults)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        # Initial step size: the box is [-5, 5]^5, so a step ~2 covers a good
        # fraction of the search space without immediately hitting the bounds.
        sigma = 2.0
        sigma_min = 1e-8
        sigma_max = 10.0

        # CMA-ES state
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Precompute C_sqrt^{-1} (inv) and C_sqrt^T (trans) via Cholesky
        def cholesky_factor(mat):
            L = np.linalg.cholesky(mat + 1e-12 * np.eye(dim))
            return L

        C_inv_sqrt = np.linalg.inv(C_sqrt)
        C_trans = C_sqrt.T

        # Mean (starting point): center of the box
        mean = np.zeros(dim)

        # Evaluate initial point
        f_mean = func(mean)
        total_calls = 1
        best_x = mean.copy()
        best_f = f_mean

        # A few extra random initial evaluations to get a better starting mean
        for _ in range(5):
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            f_try = func(x_try)
            total_calls += 1
            if f_try < best_f:
                best_x = x_try.copy()
                best_f = f_try
                mean = x_try.copy()

        # Main CMA-ES loop
        while total_calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            pop_x = np.empty((lam, dim))
            pop_f = np.empty(lam)
            for i in range(lam):
                if total_calls >= budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                pop_x[i] = y
                pop_f[i] = func(y)
                total_calls += 1

            if total_calls < budget:
                # We didn't finish the population; pad with the last valid
                # (shouldn't happen normally, but guard against it)
                pass

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_x_sorted = pop_x[order]
            pop_f_sorted = pop_f[order]

            # Update mean (weighted recombination of best mu)
            mean_new = pop_x_sorted[:mu] @ weights[:, None]

            # Update path variables
            y_inv = C_inv_sqrt @ (mean_new - mean)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_inv
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_inv

            # Update covariance
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (total_calls - 1)))
                       < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            delta_h = 1.0 - h_sigma
            C = ((1 + c_c * (2 - c_c) - c_sigma) * C
                 + c_sigma * (p_c @ p_c + (delta_h - c_sigma) * C))

            # Update sigma (CSA)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = min(max(sigma, sigma_min), sigma_max)

            # Re-orthogonalize C (ensure symmetric positive definite)
            C = (C + C.T) / 2
            try:
                C_sqrt = cholesky_factor(C)
                C_inv_sqrt = np.linalg.inv(C_sqrt)
                C_trans = C_sqrt.T
            except np.linalg.LinAlgError:
                # Fallback: reset to isotropic if Cholesky fails
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                C_trans = np.eye(dim)

            mean = mean_new

            # Track global best
            if pop_f_sorted[0] < best_f:
                best_f = pop_f_sorted[0]
                best_x = pop_x_sorted[0].copy()

            # Stagnation / restart heuristic: if sigma is very small and we
            # haven't improved in a while, do a random jump to escape a local
            # optimum. We track this with a simple counter.
            # (CMA-ES usually handles this well, but a jump can help on
            #  flat regions of f6.)
            # We'll just rely on the budget and time guard; the covariance
            # adaptation should handle most of the search.

        return best_x, best_f
# EVOLVE-BLOCK-END
