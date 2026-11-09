"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with coordinate descent refinement for Sphere."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Track calls
        calls = [0]
        max_calls = budget

        def f(x):
            if calls[0] >= max_calls:
                return float('inf')
            calls[0] += 1
            return func(x)

        # Population size for CMA-ES
        lam = 4 + int(3 * dim)  # ~19 for dim=5
        mu = lam // 2  # number of parents

        # Initialize
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        fx = f(x)
        sigma = 1.0

        # CMA-ES parameters
        cc = (4 + dim / (dim + 4)) / ((dim + 1.7) ** 2 + 4)
        cs = (dim + 1.7 + 2 / (dim + 1)) / (dim + 9 + 2 * mu / dim)
        c1 = (2 - cc) / ((dim + 1.3) ** 2 + 5)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu) / ((dim + 2) ** 2 + 5))
        mu_eff = mu / 2.0

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Update rates
        c_sigma = (dim + 1.7 + 2 / (dim + 1)) / (dim + 9 + 2 * mu / dim)
        c_cma = (4 + dim / (dim + 4)) / ((dim + 1.7) ** 2 + 4)
        c1 = (2 - c_cma) / ((dim + 1.3) ** 2 + 5)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu) / ((dim + 2) ** 2 + 5))

        # Initialize CMA-ES state
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x.copy()
        best_f = fx
        eval_count = 1

        # Main CMA-ES loop
        while eval_count < budget:
            # Check time
            if time.time() > deadline - 10:
                break

            # Generate population
            offspring = []
            offspring_f = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = f(y)
                eval_count += 1
                if calls[0] >= max_calls:
                    break
                offspring.append(y)
                offspring_f.append(fy)

            if len(offspring) < mu:
                break

            # Sort by fitness
            indices = np.argsort(offspring_f)
            offspring = [offspring[i] for i in indices]
            offspring_f = [offspring_f[i] for i in indices]

            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()

            # Compute mean of selected parents
            m_new = np.zeros(dim)
            for i in range(mu):
                m_new += weights[i] * offspring[i]

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D @ ((m_new - x) / sigma)))
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (eval_count // lam))) / chi_n) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0

            p_c = (1 - c_cma) * p_c + h_sigma * np.sqrt(c_cma * (2 - c_cma) * mu_eff) * (m_new - x) / sigma

            # Update covariance matrix
            delta_h = (1.8 + 2 / (dim + mu)) * (1 - h_sigma)
            C = ((1 - c1 - cmu + c1 * delta_h) * C + c1 * np.outer(p_c, p_c) + cmu * sum(weights[i] * np.outer(offspring[i] - x, offspring[i] - x) for i in range(mu)))

            # Update B and D via eigendecomposition (approximate for speed)
            # Use a simplified update: rescale and re-orthogonalize periodically
            if eval_count % 50 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
                C = (B @ np.diag(D ** 2) @ B.T)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 100)

            x = m_new

        # Coordinate descent refinement with remaining budget
        remaining = budget - eval_count
        if remaining > 5 and time.time() < deadline - 5:
            # Use coordinate descent on the best point
            x_cd = best_x.copy()
            fx_cd = best_f
            step = 0.5
            for _ in range(min(remaining, 100)):
                if calls[0] >= max_calls or time.time() > deadline - 2:
                    break
                improved = False
                for d in range(dim):
                    for direction in [1, -1]:
                        y = x_cd.copy()
                        y[d] += direction * step
                        y = np.clip(y, -5, 5)
                        fy = f(y)
                        if fy < fx_cd:
                            x_cd = y
                            fx_cd = fy
                            improved = True
                            if fx_cd < best_f:
                                best_f = fx_cd
                                best_x = x_cd.copy()
                    if improved:
                        break
                if not improved:
                    step *= 0.7
                else:
                    step *= 1.2
                step = np.clip(step, 0.01, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
