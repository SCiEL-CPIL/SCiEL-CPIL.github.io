"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive (1+1)-ES polish for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # safety margin before 300s kill

        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        rng = np.random.default_rng(42)

        # Phase 1: Quick exploration with random sampling to find a good basin
        n_random = min(50, max(10, self.budget // 10))
        random_points = []
        for i in range(n_random):
            if not check_time() or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_func(x)
            random_points.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort random points by fitness
        random_points.sort(key=lambda t: t[0])

        # CMA-ES implementation
        def cma_es(x0, f0, budget_remaining, max_steps):
            nonlocal best_x, best_f, calls_made

            if budget_remaining <= 0 or not check_time():
                return

            dim = self.dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(dim)))
            # Step size
            sigma = 0.5
            # Covariance matrix
            C = np.eye(dim)
            # Evolution path
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Step size control parameters
            cs = (lam // 2 + 1) / (dim + lam // 2 + 5)
            ds = 1 + 2 * max(0, (dim - 1) / (dim + 2) - 1) + cs
            c1 = (2 + 2 / (dim + 1.7)) / ((dim + 2) ** 2 + lam)
            cmu = min(2 / ((dim + 1.3) ** 2 + lam), (1 - c1) * 0.5)
            cc = (4 + dim / lam) / (dim + 4 + 2 * lam / dim)

            # Damping for cumulative covariance matrix update
            mu = lam // 2
            weights = np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights ** 2)

            x_mean = x0.copy()
            f_mean = f0

            for step in range(max_steps):
                if not check_time() or (self.budget - calls_made) <= lam:
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    if not check_time() or (self.budget - calls_made) <= lam - i:
                        break
                    z = rng.standard_normal(dim)
                    # Use Cholesky decomposition of C
                    L = np.linalg.cholesky(C)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_func(y)
                    offspring.append((fy, y))

                if len(offspring) == 0:
                    break

                # Sort offspring
                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()

                # Select best mu offspring
                mu_actual = min(mu, len(offspring))
                y_w = np.zeros(dim)
                for i in range(mu_actual):
                    y_w += weights[i] * offspring[i][1]
                y_w /= np.sum(weights[:mu_actual])

                # Update mean
                x_mean = y_w

                # Update step size
                # Compute evolution path
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (y_w - x_mean)) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (step + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))

                # Update cumulative covariance matrix evolution path
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (y_w - x_mean) / sigma

                # Update covariance matrix
                delta_d = (1 + 2 * (hsig - 1) * cc * (2 - cc)) * c1
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu_actual):
                    zi = (offspring[i][1] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)

                C = (1 + c1 * (delta_d - 1) + cmu * (mueff * (c1 + cmu) / c1 - 1)) * C + cmu * rank_mu - c1 * rank_one

                # Ensure positive definiteness
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += (1e-12 - np.min(eigvals) + 1e-9) * np.eye(dim)

                # Update step size
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt((dim + 1) / (dim - 1)) - 1))

                # Clamp sigma
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Convergence check
                if sigma < 1e-10 or np.max(np.abs(C)) < 1e-12:
                    break

        # Phase 2: Run CMA-ES from the best random point
        if random_points and best_x is not None:
            budget_for_cma = self.budget - calls_made
            max_steps = max(1, budget_for_cma // max(4, int(4 + 3 * np.log(self.dim))))
            cma_es(best_x, best_f, budget_for_cma, max_steps)

        # Phase 3: If we have budget left, try a few more restarts from different random points
        remaining = self.budget - calls_made
        if remaining > 20 and check_time():
            # Try CMA-ES from 2nd and 3rd best random points
            for idx in range(1, min(3, len(random_points))):
                if not check_time() or (self.budget - calls_made) < 10:
                    break
                x_start = random_points[idx][1].copy()
                f_start = random_points[idx][0]
                budget_for_cma = self.budget - calls_made
                max_steps = max(1, budget_for_cma // max(4, int(4 + 3 * np.log(self.dim))))
                cma_es(x_start, f_start, budget_for_cma, max_steps)

        # Phase 4: Adaptive (1+1)-ES local search with 1/5 success rule
        remaining = self.budget - calls_made
        if remaining > 0 and best_x is not None and check_time():
            sigma_local = 0.5
            x = best_x.copy()
            fx = best_f
            for _ in range(min(remaining, 200)):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break
                y = np.clip(x + sigma_local * rng.standard_normal(self.dim), -5, 5)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.5
                else:
                    sigma_local *= 0.87
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
