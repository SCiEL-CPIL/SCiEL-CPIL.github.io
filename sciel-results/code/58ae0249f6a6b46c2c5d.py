"""Optimized BBOB f18 (Schaffers F7) solver with CMA-ES and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with time-aware execution for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # 5s margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        mu = max(1, dim // 2)  # number of parents
        lam = 2 * mu           # population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        best_x = x.copy()
        best_f = np.inf
        total_calls = 0

        def time_ok():
            return time.time() < deadline

        # Generate initial population
        def generate_population():
            nonlocal total_calls
            ys = []
            for _ in range(lam):
                y = x + sigma * (B @ (D * rng.standard_normal(dim)))
                y = np.clip(y, -5, 5)
                ys.append(y)
            return ys

        # Evaluate a point and update best
        def evaluate_point(y):
            nonlocal total_calls, best_x, best_f
            fy = func(y)
            total_calls += 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
            return fy

        # Main CMA-ES loop
        while total_calls < budget and time_ok():
            # Generate and evaluate population
            ys = generate_population()
            fs = np.array([evaluate_point(y) for y in ys])
            if not time_ok():
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = np.array([ys[i] for i in idx])
            fs_sorted = fs[idx]

            # Update mean
            x_old = x.copy()
            x = sum(weights[i] * ys_sorted[i] for i in range(mu))
            x = np.clip(x, -5, 5)

            # Update sigma
            y_mean = (x_old - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update C matrix
            if np.linalg.norm(ps) / chi_n < np.sqrt((1 + 2 / (dim + 1)) * (1 - c_c)):
                pc = (1 - c_c) * pc + c_c * (x - x_old) / sigma

            delta_diag = (np.linalg.norm(ps) / chi_n > np.sqrt((1 + 2 / (dim + 1)) * (1 - c_c))) * 1
            rank_one = np.outer(pc, pc)
            rank_mu = sum(c_mu / weights[i] * np.outer(ys_sorted[i] - x_old, ys_sorted[i] - x_old) 
                         for i in range(mu))
            C = (1 - c_1 - c_mu + c_1 * delta_diag) * C + c_1 * rank_one + c_mu * rank_mu

            # Eigendecomposition
            D2, B2 = np.linalg.eigh(C)
            D2 = np.sqrt(np.maximum(D2, 1e-20))
            B = B2
            D = D2

        return best_x, best_f
# EVOLVE-BLOCK-END
