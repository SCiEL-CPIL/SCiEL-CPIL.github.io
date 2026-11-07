"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Restart CMA-ES for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0  # safety margin before 300s limit
        calls_made = 0
        best_x = None
        best_f = np.inf

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(self.dim)))  # population size
        # For dim=5: pop_size = 4 + 3*ln(5) ≈ 4+4.83 = 8 -> 8

        # Restart strategy: max iterations per restart
        max_iter_per_restart = 60
        # Stagnation threshold
        stagnation_tol = 1e-10

        # Initial sigma (step size)
        sigma_init = 2.0

        # Track progress for restart detection
        best_f_history = []
        no_improve_count = 0
        stagnation_limit = 20  # restart if no improvement for this many iterations

        total_budget = self.budget

        while calls_made < total_budget and time.time() < deadline:
            # Initialize CMA-ES state
            x_mean = rng.uniform(-5, 5, self.dim)
            # Evaluate initial point
            fx = func(x_mean)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_mean.copy()

            sigma = sigma_init
            C = np.eye(self.dim)
            # Learning rates
            mu = pop_size // 2  # number of selected individuals
            weights = np.ones(mu) / mu
            # Normalize weights
            weights = weights / np.sum(weights)
            c_sigma = (mu + 2) / (self.dim + 4)
            c_c = (4 + mu / self.dim) / (self.dim + 4 + 2 * mu / self.dim)
            c_1 = 2 / ((self.dim + 1.3) ** 2 + mu)
            c_mu = min(1 - c_1, 2 * (weights[0] - 1 + weights[1]) / ((self.dim + 2) ** 2 + mu))
            chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            best_f_this_restart = fx
            no_improve = 0
            prev_best_f = fx

            for iteration in range(max_iter_per_restart):
                if calls_made >= total_budget or time.time() > deadline:
                    break

                # Generate population
                ys = []
                fs = []
                for i in range(pop_size):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(self.dim)
                    # Apply Cholesky decomposition of C
                    L = np.linalg.cholesky(C)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    ys.append(y)
                    fs.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    if fy < best_f_this_restart:
                        best_f_this_restart = fy
                        no_improve = 0
                    else:
                        no_improve += 1

                if len(ys) == 0:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = [ys[i] for i in order]
                fs_sorted = [fs[i] for i in order]

                # Update mean
                y_w = np.zeros(self.dim)
                for i in range(mu):
                    y_w += weights[i] * ys_sorted[i]
                x_mean_new = x_mean + c_c * (y_w - x_mean)
                x_mean_new = np.clip(x_mean_new, -5, 5)

                # Update p_sigma
                d_sigma = 1 + (2 * (weights[0] - 1) / (self.dim + 2)) / (chi_n * np.sqrt(self.dim))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (y_w / sigma - p_sigma)
                # Actually, the standard formula:
                # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff)/chi_n * (y_w/sigma)
                # where mu_eff = 1/sum(w^2)
                mu_eff = 1.0 / np.sum(weights ** 2)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (y_w / sigma)

                # Update p_c
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (iteration + 1))) / chi_n < 1.4 + 2 / (self.dim + 1)
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w / sigma - p_sigma)

                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((self.dim, self.dim))
                for i in range(mu):
                    z_i = (ys_sorted[i] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(z_i, z_i)

                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                # Add small regularization if needed
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += (1e-12 - np.min(eigvals) + 1e-15) * np.eye(self.dim)

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

                x_mean = x_mean_new

                # Check for stagnation
                if no_improve >= stagnation_limit:
                    break

                if best_f_this_restart < prev_best_f:
                    prev_best_f = best_f_this_restart

            # After restart loop, check if we should restart with different initial point
            if calls_made >= total_budget or time.time() > deadline:
                break

            # Reset for next restart with a new random start
            # Optionally, bias the new start toward the best found so far
            if best_x is not None:
                # New start near best found, with some randomness
                x_mean = best_x + rng.uniform(-1, 1, self.dim)
                x_mean = np.clip(x_mean, -5, 5)
            else:
                x_mean = rng.uniform(-5, 5, self.dim)

            # Evaluate the new start
            if calls_made < total_budget and time.time() < deadline:
                fx = func(x_mean)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_mean.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
