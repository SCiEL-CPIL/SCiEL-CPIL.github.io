"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start and adaptive restarts for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        # Wrap func with deadline check
        call_count = [0]
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal best_x, best_f
            call_count[0] += 1
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial diverse sampling for early progress
        # Use 20% of budget for initial exploration
        init_calls = max(20, min(50, budget // 5))
        init_points = []
        for _ in range(init_calls):
            x = rng.uniform(-5, 5, dim)
            f = safe_func(x)
            init_points.append((f, x.copy()))

        init_points.sort(key=lambda t: t[0])
        best_f = init_points[0][0]
        best_x = init_points[0][1].copy()

        # Phase 2: CMA-ES with restarts on the best region
        # Use remaining budget
        remaining = budget - call_count[0]

        # CMA-ES parameters
        def cma_es(x0, sigma0, max_iter, budget_left):
            nonlocal best_x, best_f
            x = x0.copy()
            sigma = sigma0
            n = dim
            # CMA-ES weights
            mu = max(1, n // 2)  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            h_sigma = (1 + c_sigma / 2) * chi_n

            fx = best_f
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if budget_left[0] <= 0:
                    break

                # Generate lambda offspring
                lam = max(mu * 2, n + 1)
                if budget_left[0] < lam:
                    lam = budget_left[0]
                    if lam < 2:
                        break

                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5.0, 5.0)
                    fy = safe_func(y)
                    budget_left[0] -= 1
                    offspring.append((fy, y))

                # Sort offspring
                offspring.sort(key=lambda t: t[0])

                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * (offspring[i][1] - x)
                x = x + y_w

                # Update step size
                y_sigma = (B @ (D * (y_w / sigma))) / chi_n
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1.4 + 1 / n) * chi_n:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update covariance matrix
                y_c = B @ (D * (y_w / sigma))
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                h_c = (1 + c_c / 2) * chi_n
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = B @ (D * (offspring[i][1] - x) / sigma)
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                delta_diag = (1 - (1 - c_1 - c_mu * mu_eff / 2) ** 2) * np.diag(D ** 2)
                if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (it + 1))) < (1.4 + 1 / n) * chi_n:
                    delta_diag += (c_1 - c_mu) * (np.outer(p_c, p_c) / chi_n ** 2)
                else:
                    delta_diag += (c_1 - c_mu) * (np.outer(p_c, p_c) / chi_n ** 2 * (1 + 2 / (chi_n - 1)))

                B, D = np.linalg.eigh((1 - c_1 - c_mu) * np.eye(n) + c_1 * np.outer(p_c, p_c) + c_mu * rank_mu - delta_diag)
                D = np.sqrt(np.clip(D, 1e-20, None))

                # Track best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()

                # Stagnation check: restart if no improvement in last 50 iterations
                if it > 0 and it % 50 == 0:
                    if best_f >= fx - 1e-10:
                        break
                    fx = best_f

                # Convergence check
                if sigma < 1e-10:
                    break

            return best_x, best_f

        # Run CMA-ES from best initial point
        budget_left = [remaining]
        max_iter = 200
        best_x, best_f = cma_es(best_x, 1.0, max_iter, budget_left)

        # If budget remains, try another CMA-ES from a different good point
        if budget_left[0] > 30 and call_count[0] < budget:
            # Pick second best initial point
            if len(init_points) > 1:
                x0 = init_points[1][1].copy()
                budget_left2 = [budget_left[0]]
                best_x2, best_f2 = cma_es(x0, 1.0, max_iter, budget_left2)
                if best_f2 < best_f:
                    best_x, best_f = best_x2, best_f2

        return best_x, best_f
# EVOLVE-BLOCK-END
