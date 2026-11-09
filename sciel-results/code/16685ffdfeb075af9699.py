# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        d = self.dim
        calls_made = 0

        # Phase 1: Linear model fit (7 calls)
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        best_x = x0.copy()
        best_f = f0

        v = np.zeros(d)
        for i in range(d):
            if time.time() > deadline or calls_made >= self.budget:
                break
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            calls_made += 1
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
            v[i] = (fi - f0) / 5.0

        # Construct predicted vertex
        x_linear = np.zeros(d)
        for i in range(d):
            if v[i] > 0:
                x_linear[i] = -5.0
            elif v[i] < 0:
                x_linear[i] = 5.0
            else:
                x_linear[i] = 0.0

        if time.time() <= deadline and calls_made < self.budget:
            f_linear = func(x_linear)
            calls_made += 1
            if f_linear < best_f:
                best_f = f_linear
                best_x = x_linear.copy()

        # Phase 2: CMA-ES polish around best_x
        if time.time() > deadline or calls_made >= self.budget:
            # Stall remaining budget
            while calls_made < self.budget and time.time() <= deadline:
                func(best_x)
                calls_made += 1
            return best_x, best_f

        # CMA-ES implementation
        x = best_x.copy()
        sigma = 0.3  # Initial step size, smaller since we're near the optimum
        pop_size = max(4, int(4 + 3 * np.log(d)))  # Standard CMA-ES population size

        # Covariance matrix C
        C = np.eye(d)

        # Evolution paths
        ps = np.zeros(d)
        pc = np.zeros(d)

        # Parameters
        cs = (pop_size + 2) / (d + pop_size + 5)
        ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (d + 1)) - 1) + cs
        cmu = min(0.2, (pop_size - 2 + 1 / (d + 1)) / ((d + 2) * (pop_size + 2)))
        c_sigma = (pop_size + 1.3) / (d + pop_size + 5)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Selection weights
        mu = pop_size // 2
        w = np.log(np.arange(1, mu + 1)) - np.log(mu)
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(cmu, c_1 * (2 - c_1)) / mu_eff

        # Main CMA-ES loop
        while calls_made < self.budget and time.time() < deadline:
            # Generate population
            y = np.random.multivariate_normal(np.zeros(d), C, size=pop_size)
            x_pop = x + sigma * y
            x_pop = np.clip(x_pop, -5.0, 5.0)

            # Evaluate population
            f_pop = np.array([func(xi) for xi in x_pop])
            calls_made += pop_size

            # Update best
            idx_best = np.argmin(f_pop)
            if f_pop[idx_best] < best_f:
                best_f = f_pop[idx_best]
                best_x = x_pop[idx_best].copy()

            if calls_made >= self.budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(f_pop)
            y_sorted = y[order]
            f_sorted = f_pop[order]

            # Update mean
            x_old = x.copy()
            x = x + sigma * np.sum(w[:, np.newaxis] * y_sorted[:mu], axis=0)
            x = np.clip(x, -5.0, 5.0)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * np.dot(np.linalg.inv(np.sqrtm(C)), y_sorted[0])
            hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // pop_size)))) / chi_n
            pc = (1 - c_c) * pc + hs * c_c / np.sqrt(c_c) * (x - x_old) / sigma

            # Update covariance
            delta_hs = (1 - c_c) - c_c
            C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) - C) + c_mu * np.sum(
                w[:, np.newaxis, np.newaxis] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Update sigma
            sigma *= np.exp(c_sigma / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Check convergence
            if sigma < 1e-8:
                break

        # Stall remaining budget
        while calls_made < self.budget and time.time() <= deadline:
            func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
