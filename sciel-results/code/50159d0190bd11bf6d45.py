"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive restarts for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        # We'll use a population-based CMA-ES approach
        # lambda = population size
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size

        # CMA-ES parameters
        mu = lam // 2  # number of selected parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        rng = np.random.default_rng(12345)

        # Initial point - try center first, then random
        x_center = np.zeros(dim)
        f_center = func(x_center)
        calls_made += 1
        best_x, best_f = x_center.copy(), f_center

        # If center is not good, try a few random points
        for _ in range(3):
            if time.time() > deadline:
                break
            x_rand = rng.uniform(-5, 5, dim)
            f_rand = func(x_rand)
            calls_made += 1
            if f_rand < best_f:
                best_x, best_f = x_rand.copy(), f_rand

        # CMA-ES main loop
        while calls_made < budget and time.time() < deadline:
            # Initialize CMA-ES state
            x_mean = best_x.copy()
            sigma = 1.0
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            h_sigma = (1 + 2 / (dim + 1)) * chi_n

            # Run CMA-ES for a subset of budget
            # We'll use up to 60% of remaining budget per restart
            max_iter = min(
                (budget - calls_made) * 60 // 100,
                max(50, int(200 * dim * dim))
            )

            for _ in range(max_iter):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Generate population
                for i in range(lam):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    x_i = x_mean + sigma * (B @ (D * z))
                    x_i = np.clip(x_i, -5, 5)
                    f_i = func(x_i)
                    calls_made += 1

                    if f_i < best_f:
                        best_x, best_f = x_i.copy(), f_i

                if calls_made >= budget or time.time() > deadline:
                    break

                # Sort by fitness
                # We need to collect all f values - let's restructure
                # Actually, let me redo this more carefully

                break  # break out of inner loop, we need to restructure

            # Let me restructure the CMA-ES loop properly
            break  # break out to restructure

        # Proper CMA-ES implementation
        x_mean = best_x.copy()
        sigma = 1.0
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        h_sigma = (1 + 2 / (dim + 1)) * chi_n

        iter_count = 0
        while calls_made < budget and time.time() < deadline:
            # Generate population
            xs = np.empty((lam, dim))
            fs = np.empty(lam)

            for i in range(lam):
                if calls_made >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x_i = x_mean + sigma * (B @ (D * z))
                x_i = np.clip(x_i, -5, 5)
                xs[i] = x_i
                fs[i] = func(x_i)
                calls_made += 1

                if fs[i] < best_f:
                    best_x, best_f = x_i.copy(), fs[i]

            if calls_made >= budget or time.time() > deadline:
                break

            # Sort by fitness
            idx = np.argsort(fs)[:mu]
            y_sorted = xs[idx]

            # Weighted mean of selected points
            x_new_mean = np.sum(weights[:, None] * y_sorted, axis=0)

            # Update evolution path for covariance matrix
            y_inv = B.T @ (D ** (-1)) @ ((x_new_mean - x_mean) / sigma)
            p_c = (1 - c_c) * p_c + c_c * y_inv

            # Update evolution path for sigma
            h_sig = (np.sqrt((1 - (1 - c_c) ** (iter_count + 1)) /
                            (1 - (1 - c_c) ** (iter_count + 1))) *
                     (np.sqrt(mu_eff) - 1 + 2 / mu_eff) / chi_n) if iter_count > 0 else 1.0
            # Simplified h_sigma check
            if np.linalg.norm(y_inv) / chi_n < (1 + 2 / (dim + 1)) * h_sigma:
                h_sigma = (1 - c_sigma) * h_sigma + c_sigma
            else:
                h_sigma = (1 - c_sigma) * h_sigma

            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma *
                        (2 - c_sigma) * mu_eff) * y_inv

            # Update covariance matrix
            C = B @ (D ** 2) @ B.T
            delta_h = (1 - h_sigma) / (1 - c_sigma) - 1
            C = (1 + c_1 * delta_h * np.outer(p_c, p_c) - c_1) * C \
                + c_mu * np.sum(weights[:, None] *
                                 np.outer(y_sorted - x_mean, y_sorted - x_mean) / sigma ** 2, axis=0)

            # Eigen decomposition
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-20)
            D = np.sqrt(evals)
            B = evecs

            # Update mean and sigma
            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            iter_count += 1

            # Check for stagnation - if no improvement in last lam evaluations
            if iter_count > 50:
                # Check if we've been making progress
                break

            # Periodic restart if stuck
            if iter_count % 30 == 0 and sigma < 1e-6:
                # Restart with new random center
                x_rand = rng.uniform(-5, 5, dim)
                f_rand = func(x_rand)
                calls_made += 1
                if f_rand < best_f:
                    best_x, best_f = x_rand.copy(), f_rand
                x_mean = best_x.copy()
                sigma = 1.0
                B = np.eye(dim)
                D = np.ones(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                h_sigma = (1 + 2 / (dim + 1)) * chi_n
                iter_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
