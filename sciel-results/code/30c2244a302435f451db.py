"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using CMA-ES."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer for multimodal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 280.0  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Phase 1: Multi-start with diverse initial points
        n_starts = 10
        start_points = []
        # Center
        start_points.append(np.zeros(n))
        # Random points in [-4, 4]^n (where minimizer is guaranteed to be)
        for _ in range(n_starts - 1):
            start_points.append(rng.uniform(-4, 4, size=n))

        # Evaluate all start points
        for x0 in start_points:
            if self.calls >= budget:
                break
            if time.time() > self.deadline:
                break
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            self.calls += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x0.copy()

        if self.calls >= budget or time.time() > self.deadline:
            return self.best_x, self.best_f

        # Phase 2: CMA-ES implementation
        # Initialize CMA-ES parameters
        x_best = self.best_x.copy()
        sigma = 1.0
        pop_size = max(4, int(4 + 3 * np.log(n)))  # Population size
        C = np.eye(n)  # Covariance matrix
        ps = np.zeros(n)  # Evolution path
        cs = 1.0 / (pop_size + n + 2)  # Path length adaptation rate
        c_sigma = (1 + 2 * max(0, np.sqrt((pop_size - 1) / (n + 1)) - 1)) / (pop_size + n + 3)
        c_c = (4 + pop_size / n) / (n + 4 + 2 * pop_size / n)  # Mean adaptation rate
        c_1 = 2 / ((n + 1.3) ** 2 + pop_size)  # Covariance adaptation rate
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - n - 2) / ((n + 2) ** 2 + pop_size))
        mu = pop_size // 2  # Number of weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / (weights ** 2).sum()
        c_sigma = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)) / (pop_size + n + 3)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff / 2 + 1) - n - 2) / ((n + 2) ** 2 + mu_eff))

        # Track best
        best_x_cma = x_best.copy()
        best_f_cma = self.best_f
        stagnation_count = 0
        max_stagnation = 20  # Restarts after 20 generations without improvement

        while self.calls < budget and time.time() < self.deadline:
            if stagnation_count >= max_stagnation:
                # Restart with a new random point
                x_new = rng.uniform(-4, 4, size=n)
                fx_new = func(x_new)
                self.calls += 1
                if fx_new < self.best_f:
                    self.best_f = fx_new
                    self.best_x = x_new.copy()
                    best_x_cma = x_new.copy()
                    best_f_cma = fx_new
                    x_best = x_new.copy()
                else:
                    x_best = best_x_cma.copy()
                sigma = 1.0
                C = np.eye(n)
                ps = np.zeros(n)
                stagnation_count = 0
                continue

            # Generate population
            population = []
            fitnesses = []

            for i in range(pop_size):
                if self.calls >= budget or time.time() > self.deadline:
                    break

                # Sample from multivariate normal
                z = rng.standard_normal(n)
                # Transform using Cholesky decomposition
                L = np.linalg.cholesky(C)
                y = L @ z
                candidate = x_best + sigma * y
                candidate = np.clip(candidate, -5, 5)

                fc = func(candidate)
                self.calls += 1

                population.append(candidate)
                fitnesses.append(fc)

                if fc < self.best_f:
                    self.best_f = fc
                    self.best_x = candidate.copy()

            if not population:
                break

            # Sort by fitness
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]

            # Update best
            if fitnesses[0] < best_f_cma:
                best_f_cma = fitnesses[0]
                best_x_cma = population[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Compute new mean (weighted average of best mu individuals)
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * population[i]
            x_best = new_mean

            # Update evolution path
            y_inv = np.linalg.solve(L, x_best - np.array(population[0]))  # This is wrong, need to fix
            # Actually, we need to compute the inverse properly
            # Let's use a simpler approach for the evolution path
            diff = x_best - np.array(population[0])
            try:
                y = np.linalg.solve(L, diff)
            except:
                y = diff
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y

            # Check for restart due to no progress or ill-conditioned C
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (self.calls // pop_size + 1))) > (1.4 + 2 / (n + 1)) * np.sqrt(np.trace(C)):
                break

            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (self.calls // pop_size + 1))) - np.sqrt(2 * n / (n + 1))))
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

            # Update covariance matrix
            delta_hs = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (self.calls // pop_size + 1))) ** 2 - 1)
            C = (1 + c_1 + c_mu - c_c) * C + c_c * (np.outer(ps, ps) - C)
            for i in range(mu):
                vi = (population[i] - np.array(population[0])) / sigma
                C += c_mu * weights[i] * np.outer(vi, vi)

            # Ensure C is positive definite
            C = 0.5 * (C + C.T)
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C = C + 1e-12 * np.eye(n)

        # Phase 3: Final high-precision polish using Powell's method
        if self.calls < budget and time.time() < self.deadline and self.best_x is not None:
            try:
                result = minimize(
                    func,
                    self.best_x,
                    method='Powell',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 50, 'maxfev': budget - self.calls}
                )
                if result.fun < self.best_f:
                    self.best_f = result.fun
                    self.best_x = result.x.copy()
            except Exception:
                pass

        return self.best_x, self.best_f
