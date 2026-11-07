"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES variant with small population for BBOB-style Schwefel on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # lambda = population size
        lam = 10
        # mu = number of weights (half of lambda)
        mu = lam // 2
        # Weights: log-sum-normal distribution
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        # Step size
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Initial center and sigma
        x = np.zeros(dim)
        sigma = 3.0

        # State
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)  # For efficient sampling (approximated)
        # Actually for dim=5, we can just use C directly with Cholesky each time

        # Track best
        best_x = x.copy()
        best_f = np.inf
        calls_made = 0

        # Initial evaluation
        try:
            f0 = func(x)
            calls_made += 1
        except Exception:
            f0 = np.inf
        if f0 < best_f:
            best_f = f0
            best_x = x.copy()

        # Main loop
        while calls_made < budget:
            # Time check
            if time.time() > deadline:
                break

            # Sample offspring
            # For dim=5, Cholesky is cheap
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            # Generate lambda offspring
            offspring_x = []
            offspring_f = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = L @ z
                xi = x + sigma * y
                np.clip(xi, -5, 5, out=xi)
                offspring_x.append(xi)
                try:
                    fi = func(xi)
                    calls_made += 1
                except Exception:
                    fi = np.inf
                offspring_f.append(fi)

                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

                # Time check inside loop for safety
                if calls_made >= budget or time.time() > deadline:
                    break

            if calls_made >= budget or time.time() > deadline:
                break

            # Sort offspring by fitness
            indices = np.argsort(offspring_f)
            sorted_x = [offspring_x[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]

            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (sorted_x[i] - x)
            x_new = x + y_w

            # Update step size
            y_sigma = np.zeros(dim)
            for i in range(mu):
                y_sigma += weights[i] * (sorted_x[i] - x) / sigma

            # ps update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_sigma)
            norm_ps = np.linalg.norm(ps)

            # Update C
            # Rank-1 update
            y_w_scaled = y_w / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (y_w_scaled @ y_w_scaled.T + (1 - (1 - c_c) ** 2) * pc @ pc.T) + c_mu * sum(
                weights[i] * (sorted_x[i] - x) @ (sorted_x[i] - x).T / (sigma ** 2) for i in range(mu)
            )

            # pc update
            h_sig = 1.0 if norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - lam))) < (1 + 3 / dim) * chi_n else 0.0
            pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w_scaled

            # Update C_inv_sqrt approximation (for dim=5, just recompute Cholesky next iteration)
            # We don't need explicit C_inv_sqrt for sampling, only for ps update
            # Recompute for ps update in next iteration
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Accept new mean
            x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
