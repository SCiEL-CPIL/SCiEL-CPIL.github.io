"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Proper CMA-ES with time guard for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))  # population size
        mu = lambda_ // 2  # number of selected parents

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Step size control
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        ds = 1 + 2 * max(0.0, np.sqrt((mu_eff / (n + 1) - 1) / (n + 1)) - 1) + cs
        cc = (4 + n / mu_eff) / (n + 4 + 2 * mu_eff / (n + 1))
        c1 = (2 + 2 / (n + 1.3)) / ((n + 4) ** 2 + mu_eff)
        cmu = min(0.5, (2 - c1 + 2 / (n + 1.3)) / ((n + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        B = np.eye(n)
        D = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1

        gen = 0
        no_improve = 0
        prev_best = best_f

        while calls_made < budget and time.time() < deadline:
            gen += 1

            # Generate population
            ys = np.zeros((lambda_, n))
            fs = np.zeros(lambda_)

            for i in range(lambda_):
                if calls_made >= budget or time.time() >= deadline:
                    break
                z = rng.standard_normal(n)
                y = x_mean + sigma * (B @ (D @ z))
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls_made += 1
                ys[i] = y
                fs[i] = fy
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Check if we have enough evaluations
            if calls_made <= 1:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order[:mu]]
            fs_sorted = fs[order[:mu]]

            # Weighted recombination
            x_new = np.sum(w[:, None] * ys_sorted, axis=0)

            # Update step-size control (p_sigma)
            y_mean = (x_new - x_mean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (B.T @ y_mean)

            # h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Use a simplified h_sigma: compare to threshold
            # (1 - (1-cs)^gen) approximation for the normalization
            if gen > 100:
                norm_factor = 1.0 / np.sqrt(1.0 - (1 - cs) ** (2 * gen))
            else:
                norm_factor = 1.0
            h_sigma = (norm_p_sigma / norm_factor) < (1.4 + 2.0 / (n + 1)) * chi_n

            # Update path p_c
            p_c = (1 - cc) * p_c + h_sigma * cc * y_mean

            # Update covariance matrix C
            # Rank-one update
            C = (1 + c1 * cc * (h_sigma * (2 - cc) - 1)) * C + c1 * np.outer(p_c, p_c)
            # Rank-mu update
            for i in range(mu):
                yv = (ys_sorted[i] - x_mean) / sigma
                C += cmu * w[i] * np.outer(yv, yv)

            # Ensure C is symmetric and positive definite
            C = 0.5 * (C + C.T)

            # Eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs
                D = np.diag(np.sqrt(eigvals))
            except np.linalg.LinAlgError:
                # Fallback: reset to identity
                B = np.eye(n)
                D = np.eye(n)
                C = np.eye(n)

            # Update mean
            x_mean = x_new

            # Adapt sigma (simple 1/5 rule for sigma)
            # Count successes in this generation
            successes = np.sum(fs_sorted < fs_sorted[-1])
            if successes > mu * 0.4:
                sigma *= 1.1
            elif successes < mu * 0.2:
                sigma *= 0.9
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Track improvement
            if best_f < prev_best - 1e-10:
                no_improve = 0
            else:
                no_improve += 1
            prev_best = best_f

            # Early termination if sigma too small or no improvement
            if sigma < 1e-8 or no_improve > 50:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
