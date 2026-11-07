"""Rewritten optimizer for BBOB f6 (Attractive sector) with improved (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f6 (Attractive sector) in 5D with budget 500.
    
    Strategy:
      - Start at origin (reasonable prior since minimizer is in [-4,4]^5).
      - Use (1+1)-ES with adaptive step size based on success rate.
      - Aggressive sigma adaptation to converge quickly.
      - Time guard to ensure we finish within the 330s limit.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start at origin
        x = np.zeros(dim)
        fx = func(x)
        calls_used = 1

        # CMA-ES-lite parameters
        sigma = 2.0  # initial step size
        C = np.eye(dim)  # covariance matrix
        # Precompute Cholesky for sampling
        L = np.linalg.cholesky(C)

        # Path memory for covariance update
        p_sigma = np.zeros(dim)
        c_sigma = 1.0 / (dim + 4)
        c_c = 2 / ((dim + 1.3) ** 2 + 10)
        c_1 = 2 / ((dim + 1.7) ** 2 + 5)
        c_mu = min(c_1 * 0.5, (2 - c_1) * 0.5)
        d_sigma = 1 + 2 * max(0, np.sqrt((dim + 1) / 3 - 1) - 1)

        # Success rate tracking for sigma adaptation
        n_success = 0
        n_total = 0
        window_size = 10

        for i in range(budget - calls_used):
            # Check time
            if time.time() > deadline:
                break

            # Sample from N(x, sigma^2 * C)
            z = rng.standard_normal(dim)
            step = sigma * L @ z
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                # Success: update mean and covariance
                x_new = y
                fx_new = fy
                # Update path
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (step / sigma)
                # Sigma update
                sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) - np.sqrt(dim)))
                sigma = np.clip(sigma, 0.001, 5.0)

                # Covariance update
                s = step / sigma
                C = (1 - c_c - c_1) * C + c_c * (np.outer(p_sigma, p_sigma) + (1 - (1 - c_sigma) ** 2) * C) + c_1 * np.outer(s, s)
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If not PD, reset to identity
                    C = np.eye(dim)
                    L = np.eye(dim)

                x, fx = x_new, fx_new
                n_success += 1
            else:
                # Failure: still update path slightly
                p_sigma = (1 - c_sigma) * p_sigma

            n_total += 1

            # Adapt sigma based on success rate over the window
            if n_total >= window_size:
                success_rate = n_success / n_total
                if success_rate > 0.3:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.9
                n_success = 0
                n_total = 0

            sigma = np.clip(sigma, 0.001, 5.0)

        return x, fx
# EVOLVE-BLOCK-END
