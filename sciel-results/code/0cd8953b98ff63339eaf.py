"""CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initial population
        x0 = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        c_sigma = (n + 2) / (n + 8)
        c_c = (4 + n / n) / (n + 4 + 2 * n / n)
        mu = n // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        # Evaluate initial point
        fx = func(x0)
        best_x = x0.copy()
        best_f = fx
        calls = 1

        # CMA-ES main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            pops = []
            pop_f = []
            for i in range(mu):
                # Sample from multivariate normal
                z = rng.multivariate_normal(np.zeros(n), C, 1).flatten()
                y = x0 + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                pops.append(y)
                pop_f.append(fy)
                calls += 1
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

            if len(pops) < mu:
                break

            # Sort by fitness
            indices = np.argsort(pop_f)
            pops = [pops[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pops[0].copy()

            # Recombination
            x_new = sum(w[i] * pops[i] for i in range(mu))

            # Update covariance matrix
            diff = x_new - x0
            C = (1 - c_c) * C + c_c * (diff @ diff.T + (1 - c_c) * C - C)

            # Update step size
            ps = (1 - c_sigma) * ps + c_sigma / chi_n * (C @ diff) / sigma
            sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x0 = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
