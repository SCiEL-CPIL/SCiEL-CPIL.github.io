"""Optimized BBOB f4 (Büche-Rastrigin) solver: DE global search + CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: Differential Evolution for global exploration,
    then CMA-ES for local refinement within the best basin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        def check_time():
            return time.time() < deadline

        # --- Phase 1: Differential Evolution for global exploration ---
        # Use ~60% of budget for DE, rest for CMA-ES
        de_budget = int(budget * 0.6)
        cma_budget = budget - de_budget

        # Initialize DE population
        npop = max(10, dim * 4)  # population size
        rng = np.random.default_rng(42)

        # Clip to box
        def clip(x):
            return np.clip(x, -5, 5)

        # Initialize population
        pop = np.zeros((npop, dim))
        fpop = np.zeros(npop)
        for i in range(npop):
            pop[i] = clip(rng.uniform(-5, 5, dim))
            fpop[i] = func(pop[i])

        best_idx = np.argmin(fpop)
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]

        # DE parameters
        F = 0.8  # scale factor
        CR = 0.9  # crossover rate
        # Adaptive F: start high for exploration, decrease over time

        de_calls = npop
        max_de_calls = de_budget

        while de_calls < max_de_calls and check_time():
            # Adaptive F: decrease from 0.8 to 0.4
            progress = de_calls / max_de_calls
            F_adapt = 0.8 - 0.4 * progress
            CR_adapt = 0.9 - 0.2 * progress

            # Select best individual
            best_i = np.argmin(fpop)

            # Pick two random distinct individuals
            r1, r2 = rng.choice(npop, size=2, replace=False)
            while r1 == r2:
                r2 = rng.choice(npop)

            # Differential mutation
            v = pop[best_i] + F_adapt * (pop[r1] - pop[r2])
            v = clip(v)

            # Binary crossover
            trial = pop[best_i].copy()
            jrand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR_adapt or j == jrand:
                    trial[j] = v[j]

            trial = clip(trial)
            f_trial = func(trial)
            de_calls += 1

            # Selection
            if f_trial < fpop[best_i]:
                pop[best_i] = trial
                fpop[best_i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()

        # --- Phase 2: CMA-ES refinement around the best point found ---
        if check_time() and cma_budget > 10:
            # Initialize CMA-ES
            x_mean = best_x.copy()
            f_mean = best_f
            sigma = 1.0  # initial step size

            # Covariance matrix (identity initially)
            C = np.eye(dim)
            # Inverse and square root of C
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_isqrt = np.eye(dim)

            # Strategy parameters
            mu = max(1, npop // 2)  # number of offspring
            lambda_ = npop  # population size
            # Weights
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(mu + 0.5) - np.log(i + 1)
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # State variables
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Generate initial population to estimate
            # We'll do a simple CMA-ES loop

            cma_calls = 0
            max_cma_calls = cma_budget

            # Precompute for efficiency
            while cma_calls < max_cma_calls and check_time():
                # Generate lambda offspring
                offspring = np.zeros((lambda_, dim))
                f_offspring = np.zeros(lambda_)

                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    # Apply C_sqrt (approximate with identity for simplicity,
                    # or use Cholesky)
                    # For efficiency with small dim, use simple transformation
                    y = C_sqrt @ z
                    x_i = x_mean + sigma * y
                    x_i = clip(x_i)
                    offspring[i] = x_i
                    f_offspring[i] = func(x_i)
                    cma_calls += 1

                    if not check_time():
                        break

                if not check_time():
                    break

                # Sort by fitness
                idx = np.argsort(f_offspring)
                sorted_offspring = offspring[idx]
                sorted_f = f_offspring[idx]

                # Update best
                if sorted_f[0] < best_f:
                    best_f = sorted_f[0]
                    best_x = sorted_offspring[0].copy()

                # Update mean
                x_new = np.sum(w[:, np.newaxis] * sorted_offspring[:mu], axis=0)
                x_new = clip(x_new)

                # Update covariance
                diff = (x_new - x_mean) / sigma
                delta_sigma = 0
                if np.linalg.norm(p_sigma) > (1.4 + 1 / c_sigma) * chi_N:
                    delta_sigma = 1 - np.exp(-np.linalg.norm(p_sigma) / chi_N / c_sigma)

                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                # Update path
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_isqrt @ diff
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

                # Rank-one and rank-mu update
                C = (1 + c_1 * delta_sigma + c_mu) * C - c_1 * np.outer(diff, diff)
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (cma_calls + 1) / lambda_)) / chi_N < 1.4 + 1 / c_sigma)
                C += c_mu * (1 - h_sigma) * (
                    np.sum(w[:, np.newaxis] * (C_isqrt @ (sorted_offspring[:mu] - x_mean) / sigma).T, axis=1) *
                    (C_isqrt @ (sorted_offspring[:mu] - x_mean) / sigma)
                ) if False else C  # Simplified: skip rank-mu for speed

                # Recompute inverse and sqrt (expensive but dim=5)
                try:
                    C_inv = np.linalg.inv(C)
                    C_sqrt = np.linalg.cholesky(C)
                    C_isqrt = np.linalg.cholesky(C_inv)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_isqrt = np.eye(dim)

                x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
