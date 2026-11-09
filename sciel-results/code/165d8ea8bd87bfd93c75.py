"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with CMA-ES-like covariance adaptation and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Step size and covariance
        sigma = 1.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Adaptation parameters
        c_sigma = 1.0 / (dim + 4)
        c_c = (4 + dim / 3.0) / (dim + 12)
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * ((2 + dim / 3.0) ** 2 - dim) / ((dim + 2) ** 2 + dim))
        mu = dim // 2  # weight top mu individuals (for (1+1), mu=1 effectively)
        
        # For (1+1), we use a simplified version
        # Track the search direction
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Success rate tracking for sigma adaptation
        successes = 0
        total = 0
        success_window = 10

        for _ in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * (C_sqrt @ rng.standard_normal(dim))
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            # Update best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Adaptation
            if fy < fx:
                # Success
                successes += 1
                total += 1
                x = y
                fx = fy
                
                # Update sigma using 1/5 rule approximation
                sigma *= 1.05
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 10.0)
            else:
                total += 1
                # Failure
                sigma *= 0.95
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 10.0)

            # Update covariance (simplified for (1+1))
            if fy < fx:
                d = (y - x) / sigma
                C = (1 - c_1) * C + c_1 * (np.outer(d, d) - C)
                # Ensure positive semi-definite
                C = (C + C.T) / 2
                # Periodically recompute square root
                if calls % 10 == 0:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.clip(eigvals, 1e-10, None)
                        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    except:
                        C_sqrt = np.eye(dim)
                        C = np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
