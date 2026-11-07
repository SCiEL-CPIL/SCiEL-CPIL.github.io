"""Rastrigin optimizer using CMA-ES with multi-restart and adaptive strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """CMA-ES based optimizer for Rastrigin function in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        rng = np.random.default_rng(12345)

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            if not np.isfinite(fx):
                return np.inf
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Broad sampling to find good starting points ---
        # Sample a diverse set of points to identify promising regions
        n_samples = 30
        samples = np.empty((n_samples, dim))
        sample_vals = np.empty(n_samples)

        # Include origin and some structured points
        samples[0] = np.zeros(dim)
        for i in range(1, n_samples):
            # Mix of uniform and Gaussian-like sampling
            if i % 3 == 0:
                # Gaussian centered at 0
                samples[i] = rng.normal(0, 2, dim)
            elif i % 3 == 1:
                # Uniform in box
                samples[i] = rng.uniform(-5, 5, dim)
            else:
                # Uniform in smaller box
                samples[i] = rng.uniform(-3, 3, dim)
            samples[i] = np.clip(samples[i], -5, 5)

        for i in range(n_samples):
            if calls_made >= budget:
                break
            sample_vals[i] = counted_func(samples[i])

        # Sort by fitness
        order = np.argsort(sample_vals)
        top_k = min(5, n_samples)
        top_starts = samples[order[:top_k]]
        top_vals = sample_vals[order[:top_k]]

        # --- Phase 2: CMA-ES from multiple starting points ---
        # CMA-ES implementation (simplified, no scipy dependency for core logic)

        def cma_es(start_x, start_sigma, max_calls, rng_local):
            """Run CMA-ES from start_x for at most max_calls evaluations."""
            nonlocal calls_made, best_x, best_f

            n = dim
            pop_size = max(4, int(4 + 3 * math.log(n)))
            # Adjust pop size based on remaining budget
            max_generations = max(1, max_calls // pop_size)
            if max_generations > 100:
                max_generations = 100
                pop_size = max(4, max_calls // max_generations)

            sigma = start_sigma
            x_mean = start_x.copy()
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_sqrt_inv = np.inv(C)

            # Step size control
            c_sigma = (pop_size + 2) / (n + 4 + 2 * pop_size / n)
            d_sigma = 1 + 2 * max(0, math.sqrt((pop_size - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + pop_size / n) / (n + 4 + 3 * pop_size / n)
            c_1 = 2 / ((n + 1.3) ** 2 + pop_size)
            c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - n - 2) / ((n + 2) ** 2 + pop_size))

            # Weights for mean update
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / (weights ** 2).sum()
            c_1_eff = c_1
            c_mu_eff = c_mu

            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            best_local_x = x_mean.copy()
            best_local_f = counted_func(x_mean)
            calls_used = 1

            for gen in range(max_generations):
                if calls_used + pop_size > max_calls:
                    break
                if time.time() > deadline - 10:
                    break

                # Generate population
                ys = np.empty((pop_size, n))
                xs = np.empty((pop_size, n))
                fs = np.empty(pop_size)

                for i in range(pop_size):
                    if calls_used >= max_calls:
                        break
                    y = rng_local.multivariate_normal(np.zeros(n), C, 1)[0]
                    ys[i] = y
                    x_candidate = x_mean + sigma * y
                    x_candidate = np.clip(x_candidate, -5, 5)
                    xs[i] = x_candidate
                    fs[i] = counted_func(x_candidate)
                    calls_used += 1

                # Sort by fitness
                idx = np.argsort(fs)
                ys = ys[idx]
                xs = xs[idx]
                fs = fs[idx]

                # Update best local
                if fs[0] < best_local_f:
                    best_local_f = fs[0]
                    best_local_x = xs[0].copy()

                # Update mean
                y_w = weights[:mu] @ ys[:mu]
                x_mean_new = x_mean + sigma * y_w

                # Update step size
                p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * C_sqrt_inv @ y_w
                sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                # Update covariance
                p_c = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c) * mu_eff) * C_sqrt_inv @ y_w
                h_sigma = 1 if np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0

                C = (1 + c_1_eff + c_mu_eff - c_c) * C + \
                    c_1_eff * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + \
                    c_mu_eff * (weights[:mu] * ys[:mu] ** 2).sum() * np.zeros((n, n)) + \
                    c_mu_eff * (weights[:mu] @ np.outer(ys[:mu], ys[:mu]))

                # Ensure C is symmetric
                C = (C + C.T) / 2

                # Update square roots
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                C_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals))

                x_mean = x_mean_new

            return best_local_x, best_local_f, calls_used

        # Run CMA-ES from top starting points
        remaining_budget = budget - calls_made
        n_restarts = top_k
        calls_per_restart = max(10, remaining_budget // n_restarts)

        for i in range(n_restarts):
            if calls_made >= budget - 20:
                break
            if time.time() > deadline - 15:
                break

            start_x = top_starts[i].copy()
            # Adaptive sigma: larger for worse starts, smaller for better
            start_sigma = 1.0 + 0.5 * (top_vals[i] - top_vals[0]) / max(1, top_vals[-1] - top_vals[0] + 1e-10)
            start_sigma = max(0.5, min(start_sigma, 3.0))

            rng_local = np.random.default_rng(12345 + i * 1000)
            try:
                local_best_x, local_best_f, local_calls = cma_es(
                    start_x, start_sigma, calls_per_restart, rng_local
                )
            except Exception:
                continue

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

        # --- Phase 3: Final polishing with very tight CMA-ES ---
        if calls_made < budget - 5 and time.time() < deadline - 10:
            polish_budget = budget - calls_made
            if polish_budget > 10:
                try:
                    rng_polish = np.random.default_rng(99999)
                    polish_x, polish_f, _ = cma_es(
                        best_x.copy(), 0.1, polish_budget, rng_polish
                    )
                    if polish_f < best_f:
                        best_f = polish_f
                        best_x = polish_x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
