"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2  # number of best individuals
        # Weights (logarithmic)
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            return x, f

        # Restart CMA-ES with different initializations
        max_restarts = 3
        budget_per_restart = total_budget // max_restarts

        for restart in range(max_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            # Initial center: use best so far or random
            if best_x is not None:
                center = best_x.copy()
            else:
                center = np.random.default_rng(0 + restart * 12345).uniform(-5, 5, dim)

            sigma = 2.0
            if best_x is not None:
                sigma = 0.5  # smaller step if we have a good starting point

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            step = np.zeros(dim)

            # Generate initial population
            rng = np.random.default_rng(42 + restart * 777)

            # Evaluate initial population
            pop = []
            for i in range(lambda_):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x_i = np.clip(center + sigma * (step @ z), -5, 5)
                f_i = func(x_i)
                calls_made += 1
                pop.append((x_i, f_i))

            if not pop:
                continue

            # Sort by fitness
            pop.sort(key=lambda x: x[1])

            # Update best
            if pop[0][1] < best_f:
                best_f = pop[0][1]
                best_x = pop[0][0].copy()

            # Main loop
            while calls_made < total_budget and time.time() < deadline:
                # Select best mu
                y_w = np.zeros((mu, dim))
                for i in range(min(mu, len(pop))):
                    y_w[i] = (pop[i][0] - center) / sigma

                # Weighted mean
                m = np.sum(w[:, None] * y_w, axis=0)

                # Update covariance matrix adaptation
                # Eigen decomposition
                eigval, eigvec = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigval = np.maximum(eigval, 1e-12)

                # Compute step size adaptation
                y_sigma = np.sum(w[:, None] * y_w, axis=0)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) / chi_n) < (1.4 + 2 / (dim + 1))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update evolution path
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * m

                # Rank-one and rank-mu update
                v = p_c / np.sqrt(1 - (1 - c_c) ** 2)
                C = (1 - c_1 - c_mu + c_1 * h_sigma * dim) * C \
                    + c_1 * (v @ v.T + (1 - h_sigma) * C) \
                    + c_mu * np.sum(w[:, None, None] * (y_w[:, :, None] * y_w[:, None, :]), axis=0)

                # Symmetrize
                C = (C + C.T) / 2

                # Recompute eigendecomposition for sampling
                eigval, eigvec = np.linalg.eigh(C)
                eigval = np.maximum(eigval, 1e-12)
                B = eigvec @ np.diag(1 / np.sqrt(eigval))
                D = np.diag(np.sqrt(eigval))

                # Update center
                center = center + sigma * m

                # Generate new population
                new_pop = []
                for i in range(lambda_):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    x_i = np.clip(center + sigma * (B @ z), -5, 5)
                    f_i = func(x_i)
                    calls_made += 1
                    new_pop.append((x_i, f_i))

                if not new_pop:
                    break

                new_pop.sort(key=lambda x: x[1])

                # Update best
                if new_pop[0][1] < best_f:
                    best_f = new_pop[0][1]
                    best_x = new_pop[0][0].copy()

                pop = new_pop

                # Check for convergence
                if np.std([p[1] for p in pop]) < 1e-12:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
