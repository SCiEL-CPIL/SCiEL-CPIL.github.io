"""Optimizer for BBOB f7 (Step ellipsoidal) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Step ellipsoidal f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 280.0  # safety margin before 300s kill
        d = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial sampling to find a good starting point
        # Use a small number of random samples
        n_init = min(20, budget // 10)
        if n_init < 5:
            n_init = 5

        rng = np.random.default_rng(42)
        init_points = rng.uniform(-5, 5, size=(n_init, d))
        for i in range(n_init):
            if time.time() - start_time > timeout:
                break
            if calls >= budget:
                break
            eval_point(init_points[i])

        if best_x is None:
            x0 = rng.uniform(-5, 5, d)
            eval_point(x0)
            return best_x, best_f

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(d)))
        # Initial step size
        sigma = 2.0
        # Mean (start from best found point)
        mean = best_x.copy()

        # Covariance matrix (initially identity)
        C = np.eye(d)
        # Inverse square root of C (initially identity)
        C_sqrt_inv = np.eye(d)

        # Evolution paths
        ps = np.zeros(d)
        pc = np.zeros(d)

        # Learning rates
        mu = lam // 2  # number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))

        # Eigenvalue damping
        damps = 1.0

        # Main CMA-ES loop
        while calls < budget and time.time() - start_time < timeout:
            # Check if we have enough budget for a full population
            if calls + lam > budget:
                lam = budget - calls
                if lam < 2:
                    break
                mu = max(1, lam // 2)
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                if weights.sum() == 0:
                    weights = np.ones(mu) / mu
                else:
                    weights /= weights.sum()
                mu_eff = 1.0 / np.sum(weights**2)

            # Generate population
            population = []
            fitness = []

            for i in range(lam):
                if time.time() - start_time > timeout:
                    break
                if calls >= budget:
                    break

                # Sample from multivariate normal
                z = rng.standard_normal(d)
                y = mean + sigma * C_sqrt_inv @ z
                y = np.clip(y, -5.0, 5.0)
                fy = eval_point(y)
                population.append(y)
                fitness.append(fy)

            if len(population) < 2:
                break

            # Sort by fitness
            order = np.argsort(fitness)
            population = [population[i] for i in order]
            fitness = [fitness[i] for i in order]

            # Weighted mean of best mu individuals
            weighted_mean = np.zeros(d)
            for i in range(min(mu, len(population))):
                weighted_mean += weights[i] * population[i]
            weighted_mean /= np.sum(weights[:min(mu, len(population))])

            # Update evolution paths
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_sqrt_inv @ (weighted_mean - mean)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_sqrt_inv @ (weighted_mean - mean)

            # h_sigma condition
            norm_ps = np.linalg.norm(ps)
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
            h_sigma = norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (budget - calls))) < (1.4 + 2 / (d + 1)) * chi_n

            # pc = (1 - c_c) * pc + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (weighted_mean - mean) / sigma
            if h_sigma:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (weighted_mean - mean) / sigma
            else:
                pc = (1 - c_c) * pc

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            outer = np.outer(pc, pc)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (outer + delta_h * C)
            for i in range(min(mu, len(population))):
                y_i = (population[i] - mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            mean = weighted_mean

            # Update C_sqrt_inv (eigenvalue decomposition)
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-20, None)
                C_sqrt_inv = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            except np.linalg.LinAlgError:
                # If eigendecomposition fails, reset covariance
                C = np.eye(d)
                C_sqrt_inv = np.eye(d)
                ps = np.zeros(d)
                pc = np.zeros(d)

            # Check for convergence
            if sigma < 1e-10:
                break

            # If we're stuck (all fitness values similar), try a larger step
            if len(fitness) > 1 and max(fitness) - min(fitness) < 1e-10:
                sigma *= 2.0

        return best_x, best_f
# EVOLVE-BLOCK-END
