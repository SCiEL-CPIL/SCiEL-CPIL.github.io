"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with deadline guard for BBOB f9 (Rosenbrock rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial exploration: try several starting points
        rng = np.random.default_rng(42)

        # Try center and a few random points to get a good starting point
        candidates = [np.zeros(dim)]
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, dim))

        best_start = None
        best_start_f = np.inf
        for c in candidates:
            if call_count >= budget:
                break
            f = safe_func(c)
            if f < best_start_f:
                best_start_f = f
                best_start = c.copy()

        if best_start is None:
            best_start = np.zeros(dim)
            best_start_f = safe_func(best_start)

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        # Adjust population size based on remaining budget
        remaining = budget - call_count
        if remaining < pop_size * 10:
            pop_size = max(2, remaining // 10)

        x = best_start.copy()
        f = best_start_f
        sigma = 1.0  # initial step size

        # CMA-ES state
        C = np.eye(dim)
        C_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        cs = (pop_size + 2) / (dim + 4)
        ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs
        cc = (4 + dim / 4) / (dim + 4 + 2 * pop_size / (dim + 4))
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(1 - c1, 2 * (2 - (dim + 2) / (dim + 4)) / ((dim + 2) ** 2 + pop_size))
        chi_d = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        while call_count < budget:
            if time.time() > deadline:
                break

            # Generate population
            w = np.zeros(pop_size)
            # Logarithmic weights
            for i in range(pop_size):
                w[i] = np.log(pop_size / 2 + 0.5) - np.log(i + 1)
            w_sum = w.sum()
            w = w / w_sum
            mu_eff = 1.0 / np.sum(w[:pop_size // 2 + 1] ** 2)

            # Generate offspring
            ys = np.zeros((pop_size, dim))
            for i in range(pop_size):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (C @ z)
                ys[i] = np.clip(y, -5.0, 5.0)

            # Evaluate offspring
            fs = np.full(pop_size, np.inf)
            for i in range(pop_size):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                fs[i] = safe_func(ys[i])

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            # Update mean
            x_new = np.sum(w[:, np.newaxis] * ys_sorted, axis=0)
            x_new = np.clip(x_new, -5.0, 5.0)

            # Update covariance matrix
            y_mean = (ys_sorted - x) / sigma
            C_inv = np.linalg.inv(C + 1e-10 * np.eye(dim)) if np.any(np.abs(C) > 1e-10) else np.eye(dim)
            # For numerical stability, just use simple update
            delta_x = x_new - x
            C = (1 - c1 - cmu) * C + c1 * (np.outer(delta_x, delta_x) / sigma ** 2 + (1 - (1 - cs) ** 2) * C)
            for i in range(pop_size // 2 + 1):
                vi = (ys_sorted[i] - x) / sigma
                C += cmu * w[i] * np.outer(vi, vi)

            # Ensure C is positive semi-definite
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = np.eye(dim)

            # Update sigma
            norm_delta = np.linalg.norm(delta_x)
            if norm_delta > 1e-12:
                h_sig = min(1, (1 - (1 - cs) ** (budget - call_count)) / (1 - (1 - cs) ** 1))
                p_sigma = (1 - cs) * p_sigma + h_sig * np.sqrt(cs * (2 - cs)) * delta_x / sigma
                sigma *= np.exp(cs / chi_d * (np.linalg.norm(p_sigma) / chi_d - 1))
                sigma = np.clip(sigma, 1e-8, 50.0)

            x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
