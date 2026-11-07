"""Hybrid optimizer for BBOB f16 (Weierstrass): DE for exploration, CMA-ES for exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid DE + CMA-ES optimizer for Weierstrass function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(12345)

        # Phase 1: Differential Evolution for global exploration
        # Use about 60% of budget for DE
        de_budget = int(budget * 0.6)
        pop_size = 30  # population for DE
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate

        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        f_pop = np.array([func(x) for x in pop])
        calls_made = pop_size

        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_f = f_pop[best_idx]

        # DE loop
        while calls_made + pop_size <= de_budget and time.time() < deadline:
            new_pop = np.empty_like(pop)
            new_f = np.empty(pop_size)

            for i in range(pop_size):
                # Select three distinct random indices different from i
                while True:
                    a = rng.integers(pop_size)
                    b = rng.integers(pop_size)
                    c = rng.integers(pop_size)
                    if a != b and a != c and b != c and a != i and b != i and c != i:
                        break

                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                calls_made += 1

                if f_trial <= f_pop[i]:
                    new_pop[i] = trial
                    new_f[i] = f_trial
                else:
                    new_pop[i] = pop[i]
                    new_f[i] = f_pop[i]

            pop = new_pop
            f_pop = new_f

            # Update best
            idx = np.argmin(f_pop)
            if f_pop[idx] < best_f:
                best_f = f_pop[idx]
                best_x = pop[idx].copy()

            # Check time
            if time.time() > deadline - 30:
                break

        # Phase 2: CMA-ES for local exploitation
        cma_budget = budget - calls_made
        if cma_budget < 20:
            return best_x, best_f

        # Initialize CMA-ES from the best point found by DE
        x_mean = best_x.copy()
        sigma = 0.5  # start with smaller sigma since we're near a good point
        C = np.eye(dim)
        C_sqrt = np.eye(dim)

        # CMA-ES parameters
        lam = max(2 * dim + 2, 12)  # population size
        mu = lam // 2

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Path vectors
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Generate initial population for CMA-ES
        y = rng.standard_normal((lam, dim))
        x_pop = np.clip(x_mean + sigma * y @ C_sqrt.T, -5, 5)
        f_pop = np.array([func(x) for x in x_pop])
        calls_made += lam

        # Update best
        idx = np.argmin(f_pop)
        if f_pop[idx] < best_f:
            best_f = f_pop[idx]
            best_x = x_pop[idx].copy()

        # CMA-ES loop
        while calls_made < budget and time.time() < deadline:
            # Sort by fitness
            order = np.argsort(f_pop)
            y = y[order]
            f_pop = f_pop[order]
            y_w = y[:mu]

            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * y_w, axis=0)

            # Update path for covariance
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)

            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * C_inv_sqrt @ (x_mean - x_old) / sigma
            h_sig = 1.0 if np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (calls_made + 1)) < (1 + 2 / (dim + 1)) * chi_n else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_mean - x_old) / sigma

            # Update covariance
            C = ((1 - c_1 - c_mu) * C
                 + c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * C)
                 + c_mu * np.sum(weights[:, None, None] * np.outer(y_w, y_w), axis=0))

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 1e-12:
                C += (1e-12 - eigvals.min()) * np.eye(dim)

            # Update Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(dim)
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(dim)
                    C = np.eye(dim)

            # Generate new population
            y = rng.standard_normal((lam, dim))
            x_pop = np.clip(x_mean + sigma * y @ C_sqrt.T, -5, 5)
            f_pop = np.array([func(x) for x in x_pop])
            calls_made += lam

            # Update best
            idx = np.argmin(f_pop)
            if f_pop[idx] < best_f:
                best_f = f_pop[idx]
                best_x = x_pop[idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
