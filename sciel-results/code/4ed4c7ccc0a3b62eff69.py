"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard for BBOB f22 (21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        def time_up():
            return time.time() > deadline

        dim = self.dim
        budget = self.budget

        # Halton sequence for diverse initial points
        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result

        def halton_sequence(n, dim):
            pts = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    # Use different prime bases for each dimension
                    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                    pts[i, j] = halton(i + 1, bases[j])
            return pts

        # Generate diverse starting points
        n_starts = 8
        starts = halton_sequence(n_starts, dim)
        # Map [0,1]^dim to [-5,5]^dim
        starts = starts * 10.0 - 5.0

        # Add a few random points
        rng = np.random.default_rng(42)
        extra = rng.uniform(-5, 5, (4, dim))
        starts = np.vstack([starts, extra])
        n_starts = starts.shape[0]

        # Evaluate all starts
        best_x = None
        best_f = np.inf
        for i in range(n_starts):
            x = np.clip(starts[i], -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if time_up():
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # CMA-ES parameters
        sigma = 1.5
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2  # number of selected individuals
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initialize CMA-ES state from best point
        x_mean = best_x.copy()
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)  # We'll use Cholesky-based step
        # Use eigendecomposition for sqrt
        eigvals, eigvecs = np.linalg.eigh(C)
        C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Track budget usage
        calls_made = n_starts
        calls_left = budget - calls_made

        # CMA-ES loop
        while calls_left > 0 and not time_up():
            # Generate population
            population = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                population.append((y, None))
                calls_left -= 1
                if calls_left <= 0:
                    break

            # Evaluate population
            for i in range(len(population)):
                if time_up():
                    break
                y, _ = population[i]
                fy = func(y)
                population[i] = (y, fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(population) < 2:
                break

            # Sort by fitness
            population.sort(key=lambda x: x[1])
            selected = population[:mu]

            # New mean
            old_mean = x_mean.copy()
            x_mean_new = sum(w[i] * selected[i][0] for i in range(mu))
            x_mean_new = np.clip(x_mean_new, -5, 5)

            # Update evolution paths
            diff = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ diff
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * calls_left / (dim + 1)))) < (1.4 + 2 / (dim + 1)) * np.trace(C_sqrt_inv @ C_sqrt_inv) / dim else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (1 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(w[i] * np.outer(selected[i][0] - x_mean, selected[i][0] - x_mean) for i in range(mu))
            C = (1 + c_1 * (mu_eff - 1 - delta_h) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure positive semi-definite
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            eigvals_C = np.maximum(eigvals_C, 1e-20)
            C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T

            # Update square root
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            eigvals_C = np.maximum(eigvals_C, 1e-20)
            C_sqrt = eigvecs_C @ np.diag(np.sqrt(eigvals_C)) @ eigvecs_C.T
            C_inv_sqrt = eigvecs_C @ np.diag(1.0 / np.sqrt(eigvals_C)) @ eigvecs_C.T

            x_mean = x_mean_new

            # Adapt sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            calls_made += len(population)

            # Check if we've used all budget
            if calls_made >= budget:
                break

            # If progress is minimal, try a restart from a new random point
            if calls_made % 50 == 0:
                # Try a random restart
                if calls_left > 5:
                    x_try = rng.uniform(-5, 5, dim)
                    f_try = func(x_try)
                    calls_made += 1
                    calls_left -= 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                        x_mean = best_x.copy()
                        sigma = 1.5
                        p_sigma = np.zeros(dim)
                        p_c = np.zeros(dim)
                        C = np.eye(dim)
                        eigvals, eigvecs = np.linalg.eigh(C)
                        C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T
                        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
