"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse random search + CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse random search to find a good starting point
        # Use ~30% of budget for exploration
        n_explore = min(150, budget // 3)

        best_x = None
        best_f = np.inf

        # First, try a few structured points
        candidates = []
        # Center
        candidates.append(np.zeros(dim))
        # Corners of [-4, 4]^dim (subset)
        for _ in range(20):
            c = rng.uniform(-4, 4, dim)
            candidates.append(c)
        # Random uniform points
        for _ in range(n_explore - 21):
            candidates.append(rng.uniform(-5, 5, dim))

        for x in candidates:
            if time.time() > deadline:
                break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES refinement from the best point
        # Use remaining budget
        remaining = budget - (n_explore + 21)
        if remaining <= 10:
            return best_x, best_f

        # Implement a simple CMA-ES
        x_mean = best_x.copy()
        f_mean = best_f
        sigma = 1.0  # initial step size
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # population size

        # CMA-ES parameters
        c_sigma = (pop_size + 2) / (dim + pop_size + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) / (dim + 2) - 1) / ((dim + 2) ** 2 + pop_size))

        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (initially identity)
        C_inv_sqrt = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Weights
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff / 2 + 1) / (dim + 2) - 1) / ((dim + 2) ** 2 + mu_eff))

        calls_used = n_explore + 21
        calls_remaining = budget - calls_used

        # Track best
        global_best_x = best_x.copy()
        global_best_f = best_f

        for gen in range(calls_remaining // pop_size):
            if time.time() > deadline:
                break

            # Generate population
            archive = []
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                y = x_mean + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                archive.append((y, fy))

            if len(archive) < pop_size:
                break

            # Sort by fitness
            archive.sort(key=lambda t: t[1])

            # Update best
            if archive[0][1] < global_best_f:
                global_best_f = archive[0][1]
                global_best_x = archive[0][0].copy()

            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(min(mu, len(archive))):
                x_new_mean += weights[i] * archive[i][0]
            x_mean = x_new_mean

            # Update sigma
            sigma_old = sigma
            # Recalculate C_sqrt and C_inv_sqrt periodically (simplified)
            # For simplicity, we'll use a simplified update

            # Update evolution path for sigma
            if sigma_old > 1e-12:
                diff = (x_new_mean - x_mean_old) / sigma_old if hasattr(self, '_last_mean') else np.zeros(dim)
                # Simplified: just adapt sigma based on success rate
                n_success = sum(1 for i in range(min(mu, len(archive))) if archive[i][1] < f_mean)
                if n_success > mu * 0.5:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 10.0)

            f_mean = archive[0][1]

            # Update covariance (simplified - just track direction)
            # Full CMA-ES would update C, but for simplicity we skip detailed updates
            # The random search + local refinement should be sufficient

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
