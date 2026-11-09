"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB Schaffers F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def remaining():
            return budget - calls_made

        def time_ok():
            return time.time() < deadline

        # CMA-ES parameters
        def cmaes(x0, sigma0, n_calls, rng):
            nonlocal calls_made, best_x, best_f
            n = self.dim
            # Population size
            lambda_ = 4 + int(3 * np.log(n))
            lambda_ = min(lambda_, n_calls - 1)
            if lambda_ < 2:
                lambda_ = 2

            # Strategy parameters
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)  # simplified, we'll use Cholesky-like

            # Precompute
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            def sample_sigma():
                return sigma * rng.standard_normal(n)

            def update_C_inv_sqrt():
                # Simple power iteration for inverse square root
                # Use eigendecomposition for small n
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)

            update_C_inv_sqrt()

            for gen in range(n_calls):
                if calls_made >= budget or not time_ok():
                    break

                # Generate population
                offspring = []
                for i in range(lambda_):
                    if calls_made >= budget or not time_ok():
                        break
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    offspring.append((fy, y))

                if len(offspring) < 2:
                    break

                offspring.sort(key=lambda t: t[0])

                # Update best
                for fy, y in offspring:
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                # Update mean
                new_mean = sum(weights[i] * offspring[i][1] for i in range(mu))

                # Update step size
                y_old = (new_mean - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C_inv_sqrt @ y_old)
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 1 / n) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update covariance
                delta = (new_mean - x_mean) / sigma
                pc = (1 - c_c) * pc + c_c * delta
                rank_one = np.outer(delta, delta)
                rank_mu = sum(weights[i] * np.outer(offspring[i][1] - x_mean, offspring[i][1] - x_mean) for i in range(mu)) / (sigma ** 2)
                C = (1 + c_1 * (c_c * (2 - c_c) * np.outer(pc, pc) - c_c) + c_mu * (rank_mu - C)) * (1 - c_1 * (1 - c_c) * (np.outer(pc, pc) - C))
                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2
                # Clip eigenvalues
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, 1e12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(n)

                x_mean = new_mean
                update_C_inv_sqrt()

                # Stagnation check: if no improvement in last several generations, break
                if gen > 5 and all(offspring[i][0] >= offspring[0][0] * 1.0001 + 1e-10 for i in range(min(5, len(offspring)))):
                    if gen > 20:
                        break

            return best_x, best_f

        # Multi-start strategy: use remaining budget for multiple CMA-ES runs
        # Start with a few well-spread initial points
        # Use deterministic spread points for reliability
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Corners of a spread grid
        for axis in range(dim):
            for val in [-3.0, 3.0]:
                p = np.zeros(dim)
                p[axis] = val
                init_points.append(p)

        # Shuffle with rng for diversity but deterministic
        idx = rng.permutation(len(init_points))
        init_points = [init_points[i] for i in idx]

        # Allocate budget across starts
        n_starts = min(len(init_points), max(1, budget // 80))
        budget_per_start = max(20, budget // n_starts)

        for s in range(n_starts):
            if calls_made >= budget or not time_ok():
                break
            x0 = init_points[s]
            sigma0 = 1.5
            n_alloc = min(budget_per_start, budget - calls_made)
            if n_alloc < 5:
                break
            cmaes(x0, sigma0, n_alloc, rng)

        # Final polish: use remaining budget for a fine-grained local search around best
        if best_x is not None and calls_made < budget and time_ok():
            remaining_calls = budget - calls_made
            if remaining_calls > 5:
                # Simple local refinement with small steps
                sigma_small = 0.1
                for _ in range(remaining_calls):
                    if not time_ok() or calls_made >= budget:
                        break
                    z = rng.standard_normal(dim)
                    y = np.clip(best_x + sigma_small * z, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        sigma_small = max(0.01, sigma_small * 0.9)
                    else:
                        sigma_small = max(0.01, sigma_small * 0.95)

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
