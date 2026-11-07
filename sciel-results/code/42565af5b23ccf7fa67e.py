"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive sigma and deadline guard for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # safety margin before 300s limit

        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        # Initial point: random in the box
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        sigma = 1.0
        c1 = (2 + 2 / (n + 4)) / ((n + 2) ** 2 + 2)
        cmu = min(0.04 + c1, 0.16)
        cs = (n + 1.7) / (n + 9 + 2 * cmu)
        ds = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Covariance matrix
        C = np.eye(n)
        # Eigen decomposition for updating C
        D = np.ones(n)
        B = np.eye(n)

        # Evolution path
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        # Weights: log-linear
        w = np.log(lam) - np.log(np.arange(1, lam + 1))
        w /= w.sum()
        mu = int(np.floor(lam / 2))
        w = w[:mu]
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Track number of function calls
        calls = 1

        # Main loop
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            offspring_f = []

            for i in range(lam):
                # Sample from N(x, sigma^2 * C)
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append(y)
                offspring_f.append(fy)

                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

                # Deadline check inside population generation
                if time.time() > deadline:
                    break

            if len(offspring) < mu:
                break

            # Sort by fitness
            idx = np.argsort(offspring_f[:len(offspring)])
            sorted_offspring = [offspring[i] for i in idx[:mu]]
            sorted_f = [offspring_f[i] for i in idx[:mu]]

            # Update mean
            x_old = x.copy()
            x = sum(w[i] * sorted_offspring[i] for i in range(mu))

            # Update evolution path p_sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (D ** (-1)) @ ((x - x_old) / sigma))

            # Norm of p_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)

            # Update evolution path p_c
            h_sigma = 1.0 if (norm_p_sigma / np.sqrt(1 - (1 - cs) ** (2 * (calls - 1))) <
                              (1.4 + 2 / (n + 1)) * chi_n) else 0.0

            p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (x - x_old) / sigma

            # Update covariance matrix
            C = (1 + cs * ds * (h_sigma - 1) - c1) * C
            for i in range(mu):
                y_i = (x_old - sorted_offspring[i]) / sigma
                C += c1 * (y_i @ y_i.T)

            # Ensure positive definiteness via eigenvalue clipping
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / ds * (norm_p_sigma / chi_n - 1))

            # Update B and D from C
            D2, B2 = np.linalg.eigh(C)
            D2 = np.sqrt(np.maximum(D2, 1e-12))
            B = B2
            D = D2

            # Clip x to bounds
            x = np.clip(x, -5, 5)

        return best_x, best_fx
# EVOLVE-BLOCK-END
