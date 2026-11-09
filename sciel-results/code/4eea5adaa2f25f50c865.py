"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + (4 + dim / (dim + 4)))
        cs = (dim + 1.7) / (dim + 9)
        c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - cc - cs)
        cmu = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (min(1 - c1, 2 * (mu_eff - 2) / (mu_eff + 1)))

        # Initial setup
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        call_count = 0

        def evaluate(x_eval):
            nonlocal call_count
            val = func(x_eval)
            call_count += 1
            return val

        # Initial population
        best_x = x.copy()
        best_f = evaluate(x)

        # Generate initial population
        pop = np.zeros((lambda_, dim))
        pop_f = np.zeros(lambda_)
        for i in range(lambda_):
            pop[i] = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            pop_f[i] = evaluate(pop[i])
            if pop_f[i] < best_f:
                best_f = pop_f[i]
                best_x = pop[i].copy()

        # Sort by fitness
        order = np.argsort(pop_f)
        pop = pop[order]
        pop_f = pop_f[order]

        # Main CMA-ES loop
        while call_count < budget:
            if time.time() > deadline:
                break

            # Check if we can complete a full generation
            if call_count + lambda_ > budget:
                # Do partial generation or single steps
                for _ in range(budget - call_count):
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    # Simple step
                    if fy < pop_f[0]:
                        x = y.copy()
                    else:
                        sigma *= 0.95
                break

            # Generate new population
            pop = np.zeros((lambda_, dim))
            pop_f = np.zeros(lambda_)
            for i in range(lambda_):
                pop[i] = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                pop_f[i] = evaluate(pop[i])
                if pop_f[i] < best_f:
                    best_f = pop_f[i]
                    best_x = pop[i].copy()

            # Sort
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]

            # Update mean
            m_old = x.copy()
            x = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update step size
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (C @ (x - m_old))
            sigma *= np.exp(cs / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (call_count // lambda_ + 1))) < (1 + 1.4 / dim) * chi_n else 0.0

            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * (x - m_old) / sigma

            delta_h = (1 - h_sigma) * cc * (2 - cc)
            C = (1 + c1 * delta_h - c1 - cmu) * C \
                + c1 * np.outer(p_c, p_c) \
                + cmu / sigma ** 2 * np.sum(weights[:, None, None] * np.outer(pop[:mu] - m_old, pop[:mu] - m_old), axis=0)

            # Re-derive B and D from C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        return best_x, best_f
# EVOLVE-BLOCK-END
