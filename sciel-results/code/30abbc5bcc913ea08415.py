"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-like optimizer with adaptive covariance for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Initialize with a reasonably wide step
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive covariance: start isotropic, adapt to accepted moves
        C = np.eye(dim)
        sigma = 3.0

        # History of accepted steps for covariance estimation
        accepted_steps = []
        max_history = 10

        call_count = 1

        for i in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate candidate using current covariance
            # Cholesky decomposition for correlated steps
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            z = rng.standard_normal(dim)
            step = sigma * L @ z
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                # Accept and update
                actual_step = y - x
                x = y
                fx = fy

                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

                # Update covariance based on accepted step
                accepted_steps.append(actual_step)
                if len(accepted_steps) > max_history:
                    accepted_steps.pop(0)

                if len(accepted_steps) >= 3:
                    # Estimate covariance from recent accepted steps
                    # Normalize steps to unit length first, then compute outer products
                    norms = np.linalg.norm(accepted_steps, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-10)
                    normalized = accepted_steps / norms
                    # Weighted average of outer products, with recent steps weighted more
                    weights = np.linspace(0.5, 1.0, len(accepted_steps))
                    weights /= weights.sum()
                    C_new = np.zeros((dim, dim))
                    for w, s in zip(weights, normalized):
                        C_new += w * np.outer(s, s)
                    # Ensure positive definite
                    C = 0.7 * C + 0.3 * C_new
                    # Normalize trace to keep scale controlled
                    trace = np.trace(C)
                    if trace > 1e-10:
                        C = C / trace * dim

                # Increase sigma on success
                sigma *= 1.3
                if sigma > 8.0:
                    sigma = 8.0
            else:
                # Reject
                sigma *= 0.8
                if sigma < 0.01:
                    sigma = 0.01

                # If stuck for a while, try a larger exploration step
                if i % 20 == 19:
                    # Occasional restart-like exploration
                    z_large = rng.standard_normal(dim) * 2.0
                    y_explore = np.clip(x + z_large, -5, 5)
                    fy_explore = func(y_explore)
                    call_count += 1
                    if fy_explore < fx:
                        x = y_explore
                        fx = fy_explore
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        sigma *= 1.5

        return best_x, best_fx
# EVOLVE-BLOCK-END
