"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with covariance adaptation and time guard for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # Start at origin: shift is in [-4,4]^5, so origin is a good prior
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Step size and covariance
        sigma = 1.5
        C = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Adaptation parameters for covariance
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)

        # 1/5 success rule parameters
        success_window = 10
        successes = 0
        total = 0

        while calls < budget:
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

                # Update covariance (simplified for (1+1))
                d = (y - x + sigma * (C_sqrt @ rng.standard_normal(dim)) - x) / sigma
                # Actually, d should be the step direction in the standardized space
                # Let's recompute: the step was sigma * C_sqrt @ z, so z = C_sqrt^-1 @ (y - x) / sigma
                # But since y = x + sigma * C_sqrt @ z, we have z = C_sqrt^-1 @ (y-x) / sigma
                # For simplicity, use the actual step direction
                step = (y - (y - sigma * (C_sqrt @ (C_sqrt.T @ np.linalg.solve(C, (y - x) / sigma))))) / sigma
                # This is getting complicated. Let's use a simpler approach:
                # The direction in the transformed space is z, but we don't store it.
                # Instead, use the actual step direction normalized
                step_dir = (y - (y - sigma * (C_sqrt @ (C_sqrt.T @ np.linalg.solve(C, (y - x) / sigma))))) / sigma
                # Simpler: just use the raw step
                raw_step = (y - x) / sigma  # This is C_sqrt @ z
                # We want to update C in the standard normal space
                # The direction in standard normal space is z = C_sqrt^-1 @ raw_step
                try:
                    z = np.linalg.solve(C_sqrt, raw_step)
                except:
                    z = raw_step
                C = (1 - c_1) * C + c_1 * (np.outer(z, z) - C)
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

                # Adapt sigma using 1/5 rule
                if total >= success_window:
                    success_rate = successes / total
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.9
                    successes = 0
                    total = 0
            else:
                total += 1
                if total >= success_window:
                    success_rate = successes / total
                    if success_rate < 0.2:
                        sigma *= 0.9
                    successes = 0
                    total = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)

            # If stuck, try a restart
            if sigma < 1e-5:
                x = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_x, best_f = x.copy(), fx
                sigma = 1.0
                C = np.eye(dim)
                C_sqrt = np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
