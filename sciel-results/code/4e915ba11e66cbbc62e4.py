"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart 1+1-CMA-ES with deadline guard for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Multi-restart strategy: several shorter runs from different starts
        # to escape local minima in the 21-peak landscape
        n_restarts = 5
        budget_per_restart = max(1, self.budget // n_restarts)

        best_x = None
        best_f = np.inf

        for r in range(n_restarts):
            if time.time() > deadline:
                break

            # Different starting point for each restart
            if r == 0:
                x = rng.uniform(-5, 5, self.dim)
            else:
                # Start near best found so far but with perturbation
                if best_x is not None:
                    x = np.clip(best_x + rng.normal(0, 2.0, self.dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, self.dim)

            fx = func(x)
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # 1+1-CMA-ES parameters
            sigma = 1.5
            C = np.eye(self.dim)
            pc = np.zeros(self.dim)
            cs = 1 + 2 * max(0, (self.dim - 1) / (self.dim + 2) - 1)
            c1 = (2 + 2 * ((self.dim + 1) / (self.dim + 3)) ** 2) / ((self.dim + 2) ** 2 + 4)
            cmu = min(c1 * 3 / 2, (2 - c1 + 2 / (self.dim + 1)) * 0.5)
            mu = 1  # for 1+1, mu=1

            for step in range(budget_per_restart):
                if time.time() > deadline:
                    break

                # Sample direction
                d = rng.standard_normal(self.dim)
                # Transform by C^(1/2) approximation using Cholesky
                try:
                    L = np.linalg.cholesky(C)
                    z = L @ d
                except np.linalg.LinAlgError:
                    z = d

                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)

                # Update path
                if fy < fx:
                    # Success
                    x_new = y
                    fx_new = fy
                    success = True
                else:
                    x_new = x
                    fx_new = fx
                    success = False

                # Update evolution path
                pc = (1 - cs) * pc + cs * np.sign(fy < fx) * z / np.linalg.norm(z + 1e-300)

                # Update covariance
                if success:
                    rank_one = np.outer(z, z)
                    C = (1 + c1 + cmu - c1) * C + c1 * (rank_one - C)
                else:
                    C = (1 - c1) * C

                # Ensure positive semi-definite
                C = 0.5 * (C + C.T)
                # Regularize if needed
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.min() < 1e-8:
                    C += (1e-8 - eigvals.min()) * np.eye(self.dim)

                # Update sigma with 1/5 rule approximation
                if success:
                    sigma *= 1.2
                else:
                    sigma *= 0.9

                sigma = max(sigma, 1e-4)
                sigma = min(sigma, 10.0)

                x = x_new
                fx = fx_new

                if fx < best_f:
                    best_f, best_x = fx, x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
