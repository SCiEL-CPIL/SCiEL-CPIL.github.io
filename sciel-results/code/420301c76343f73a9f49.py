"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with sigma adaptation for Bent Cigar optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Sobol initialization to find a good starting point
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        num_starts = 16
        m = 1
        while 2**m < num_starts:
            m += 1
        sobol_samples = sampler.random(2**m)
        sobol_scaled = -5 + 10 * sobol_samples[:num_starts]

        # Evaluate initial points to find best start
        best_x = None
        best_f = np.inf
        calls = 0
        for i in range(num_starts):
            if calls >= budget or time.time() > deadline:
                break
            x = sobol_scaled[i].copy()
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # If no points evaluated, use center
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # (1+1)-ES with sigma adaptation
        x = best_x.copy()
        f = best_f
        sigma = 1.0
        remaining = budget - calls

        # Covariance matrix adaptation for anisotropic landscape
        C = np.eye(dim)
        pc = np.zeros(dim)
        cs = 1.0 / (dim + 3)
        c1 = 2 / ((dim + 1.3) ** 2 + 1)
        cmu = min(0.5 - c1, 2 / ((dim + 2) ** 2 + 5))

        # Success rate based sigma adaptation
        success_rate = 0.35

        while calls < budget and time.time() < deadline:
            # Generate candidate
            z = np.random.randn(dim)
            # Transform using C (Cholesky)
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            y = L @ z
            y_norm = np.linalg.norm(y)
            if y_norm > 1e-10:
                y = y / y_norm * np.linalg.norm(z)  # normalize then rescale

            x_trial = x + sigma * y
            x_trial = np.clip(x_trial, -5, 5)
            f_trial = func(x_trial)
            calls += 1

            # Update memory
            pc = (1 - cs) * pc + cs * (y / sigma if sigma > 1e-10 else np.zeros(dim))

            if f_trial < f:
                # Success
                f = f_trial
                x = x_trial.copy()
                # Increase sigma
                sigma *= 1.5
                # Update C
                delta = y / sigma if sigma > 1e-10 else np.zeros(dim)
                C = (1 + c1 + cmu) * C + c1 * (2 - c1) * np.outer(pc, pc) - (1 + c1 + cmu) * np.eye(dim)
                # Ensure positive definite
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.min() < 1e-10:
                    C = C + (1e-10 - eigvals.min()) * np.eye(dim)
            else:
                # Failure
                # Decrease sigma
                sigma *= 0.87
                # Lighter update to C
                delta = -y / sigma if sigma > 1e-10 else np.zeros(dim)
                C = (1 + c1 * 0.5) * C - c1 * 0.5 * np.outer(pc, pc)
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.min() < 1e-10:
                    C = C + (1e-10 - eigvals.min()) * np.eye(dim)

            # Clamp sigma
            sigma = max(1e-6, min(sigma, 10.0))

        return x, f
# EVOLVE-BLOCK-END
