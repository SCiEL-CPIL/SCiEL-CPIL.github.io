"""Optimizer for BBOB f12 (Bent cigar) with CMA-ES and time guard."""
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time budget guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        sigma = 4.0
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # CMA-ES state
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))

        # Adaptation parameters
        c_sigma = (pop_size + 2) / (dim + pop_size + 5)
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2.0 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        gen = 0
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            pop = []
            for i in range(pop_size):
                if time.time() > deadline or calls >= budget:
                    break
                z = rng.standard_normal(dim)
                direction = C @ z
                y = np.clip(x + sigma * direction, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                pop.append((fy, y.copy()))

            if len(pop) < mu:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[0])
            selected = [pop[i][1] for i in range(mu)]

            # Weighted mean
            m_new = np.zeros(dim)
            for i in range(mu):
                m_new += weights[i] * selected[i]

            # Update evolution paths
            diff = m_new - x
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ diff) / sigma
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) / chi_n
            h_sigma = min(h_sigma, 1.4 + 2.0 / (dim + 1))

            if h_sigma < 1.4 + 2.0 / (dim + 1):
                p_c = (1 - c_c) * p_c + c_c * diff / sigma

            # Update covariance matrix
            C = ((1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - c_c) * (1 - 2 * c_c) * np.eye(dim)))
            outer_sum = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = selected[i] - x
                outer_sum += np.outer(diff_i, diff_i)
            C += (c_mu / sigma**2) * outer_sum

            # Symmetrize
            C = 0.5 * (C + C.T)
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update mean
            x = m_new
            x = np.clip(x, -5, 5)

            # Evaluate new center if budget allows
            if calls < budget and time.time() < deadline:
                fx = func(x)
                calls += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

            gen += 1

        return best_x, best_fx
