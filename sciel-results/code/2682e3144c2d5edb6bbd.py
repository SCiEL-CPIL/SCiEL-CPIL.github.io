"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time-bounded evaluation and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial exploration with scattered points
        # Use a small set of well-distributed starting points
        rng = np.random.default_rng(42)

        # Evaluate a few diverse starting points
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Random points
        for _ in range(min(4, budget - 1)):
            init_points.append(rng.uniform(-4, 4, dim))
        # Axis-aligned points
        for i in range(dim):
            if total_calls < budget - 1:
                p = np.zeros(dim)
                p[i] = 3.0
                init_points.append(p)

        for p in init_points:
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break
            p = np.clip(p, -5, 5)
            fp = func(p)
            total_calls += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        # Phase 2: CMA-ES from the best starting point
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        pop_size = min(pop_size, budget - total_calls)
        if pop_size < 2:
            # Not enough budget for CMA-ES, just do random search
            for _ in range(budget - total_calls):
                if time.time() > deadline:
                    break
                y = rng.uniform(-5, 5, dim)
                fy = func(y)
                total_calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            return best_x, best_f

        # CMA-ES state
        x_mean = best_x.copy()
        sigma = 1.0
        # Covariance matrix
        C = np.eye(dim)
        # Inverse of C for efficiency
        C_inv = np.eye(dim)

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        c_sigma = 1.0 / (pop_size + 3)
        c_c = (2 + 2.0 / (dim + 1)) / (pop_size + 2)
        mu = pop_size // 2
        w = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / (w @ w)
        c_sigma = 1.0 / (pop_size + 3)
        c_c = (2 + 2.0 / (dim + 1)) / (pop_size + 2)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (pop_size + mu_eff - 1)) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Track history for stagnation detection
        no_improve_count = 0
        stagnation_threshold = 50

        while total_calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            offspring_f = []
            for i in range(pop_size):
                if total_calls >= budget:
                    break
                if time.time() > deadline:
                    break
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                offspring.append(y)
                offspring_f.append(fy)

            if not offspring:
                break

            # Sort by fitness
            indices = np.argsort(offspring_f)
            offspring = [offspring[i] for i in indices]
            offspring_f = [offspring_f[i] for i in indices]

            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Compute weighted mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * offspring[i]

            # Update evolution paths
            y_sigma = (x_new - x_mean) / (sigma * chi_n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            C_sqrt = np.linalg.cholesky(C)
            C_sqrt_inv = np.linalg.inv(C_sqrt)
            y_c = C_sqrt_inv @ (x_new - x_mean)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

            # Update covariance matrix
            outer = np.outer(p_c, p_c)
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (total_calls // pop_size + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            C = (1 + c_c * (2 - c_c) + c_c * mu_eff * h_sigma * (np.outer(p_c, p_c) - C)) * C
            for i in range(mu):
                z = C_sqrt_inv @ (offspring[i] - x_mean)
                C += w[i] * (np.outer(z, z) - C)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            except:
                C = np.eye(dim)
                C_inv = np.eye(dim)

            # Update mean
            x_mean = x_new

            # Stagnation detection: if no improvement for many generations, restart
            if no_improve_count >= stagnation_threshold and total_calls < budget - 20:
                # Restart from a new random point
                no_improve_count = 0
                sigma = 2.0
                C = np.eye(dim)
                C_inv = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                # Try a few random restarts
                for _ in range(min(5, budget - total_calls)):
                    if time.time() > deadline:
                        break
                    y = rng.uniform(-4, 4, dim)
                    fy = func(y)
                    total_calls += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                x_mean = best_x.copy()

            # Adaptive sigma adjustment based on progress
            if no_improve_count > 20:
                sigma *= 0.7
            elif no_improve_count == 0:
                sigma *= 1.1

            sigma = np.clip(sigma, 0.01, 10.0)

        # Final local refinement around best point
        if total_calls < budget:
            local_sigma = 0.1
            for _ in range(min(50, budget - total_calls)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    local_sigma *= 1.2
                else:
                    local_sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
