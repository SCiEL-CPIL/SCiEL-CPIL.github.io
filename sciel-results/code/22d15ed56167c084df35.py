"""Optimized black-box optimizer for BBOB-style Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with multi-restart for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        c_mu = 2 / ((dim + 1.3) ** 2 + mu_eff) if False else 2 / ((dim + 1.3) ** 2 + sum(weights[:mu]))
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu_ = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2.2) ** 2 + mu_eff)
        mu_eff = sum(weights[:mu]) ** 2

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial sampling to find a good starting region
        n_initial = 15
        initial_points = []
        for i in range(n_initial):
            if calls >= budget or time.time() > deadline:
                break
            # Mix of structured and random points
            if i < 5:
                # Deterministic spread
                t = (i + 0.5) / 5
                x = np.array([(-5 + 10 * t) if j == 0 else rng.uniform(-5, 5) for j in range(dim)])
            else:
                x = rng.uniform(-5, 5, dim)
            x = np.clip(x, -5, 5)
            fx = evaluate(x)
            initial_points.append((fx, x.copy()))

        initial_points.sort(key=lambda t: t[0])

        # Phase 2: Run CMA-ES from the best initial point
        # Allow multiple restarts if stuck
        max_restarts = 3
        restart_threshold = 200  # no improvement in this many calls triggers restart

        for restart in range(max_restarts):
            if calls >= budget or time.time() > deadline:
                break

            # Initialize CMA-ES state
            if restart == 0:
                mean = initial_points[0][1].copy()
            else:
                # Restart from a different initial point or random
                if restart < len(initial_points):
                    mean = initial_points[restart][1].copy()
                else:
                    mean = rng.uniform(-5, 5, dim)
                mean = np.clip(mean, -5, 5)

            sigma = 2.0  # initial step size
            C = np.eye(dim)  # covariance matrix
            pc = np.zeros(dim)  # evolution path for covariance
            ps = np.zeros(dim)  # evolution path for sigma
            cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2.2) ** 2 + mu_eff)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Precompute matrix square root
            eigvals, eigvecs = np.linalg.eigh(C)
            C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T

            calls_at_start = calls
            no_improve_count = 0
            last_best_f = best_f

            while calls < budget:
                if time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    x_i = mean + sigma * (C_sqrt @ z)
                    x_i = np.clip(x_i, -5, 5)
                    offspring.append(x_i)

                # Evaluate offspring
                offspring_f = []
                for x_i in offspring:
                    if calls >= budget or time.time() > deadline:
                        break
                    fx = evaluate(x_i)
                    offspring_f.append(fx)

                if len(offspring_f) < lambda_:
                    break

                # Sort by fitness
                order = np.argsort(offspring_f)
                offspring_sorted = [offspring[i] for i in order]
                offspring_f_sorted = [offspring_f[i] for i in order]

                # Update best
                if offspring_f_sorted[0] < best_f:
                    best_f = offspring_f_sorted[0]
                    best_x = offspring_sorted[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += lambda_

                # Update mean
                new_mean = sum(weights[i] * offspring_sorted[i] for i in range(mu))

                # Update evolution paths
                y = (new_mean - mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls - calls_at_start + 1)) <
                        (1 + 2 / (dim + 1)) * chi_n)
                pc = (1 - cc) * pc + (hsig / cc) * y

                # Update covariance
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                C = (1 + c1 + cmu - c1 - cc) * C
                C += c1 * (np.outer(pc, pc) - C)
                C += cmu * sum(weights[i] * np.outer(
                    (offspring_sorted[i] - mean) / sigma,
                    (offspring_sorted[i] - mean) / sigma
                ) for i in range(mu))

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

                # Update sigma
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update mean
                mean = new_mean
                mean = np.clip(mean, -5, 5)

                # Check for stagnation
                if no_improve_count > restart_threshold:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
