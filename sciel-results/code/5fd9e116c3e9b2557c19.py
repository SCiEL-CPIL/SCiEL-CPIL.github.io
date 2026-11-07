"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population for Bent Cigar (f12) optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lam // 2  # number of selected parents
        # Weights (logarithmic)
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_cov_pos = (2 + 2 * mu_eff / n) / (n + 3 + 2 * mu_eff / n)
        c_cov_neg = (2 + mu_eff / n) / (n + 3 + 2 * mu_eff / n)

        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)  # C^{-1/2}
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Track best
        best_x = x_mean.copy()
        best_fx = func(x_mean)
        calls_used = 1

        # Initial covariance matrix square root (Cholesky)
        D, B = np.linalg.eigh(C)
        D = np.maximum(D, 1e-12)
        C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

        # Stagnation tracking
        last_improvement_gen = 0
        stagnation_threshold = 15  # generations without improvement
        no_improve_count = 0

        for gen in range(1, 2000):
            if time.time() > deadline:
                break
            if calls_used + lam > budget:
                # Use remaining calls
                rem = budget - calls_used
                if rem < 1:
                    break
                # Just do a few single evaluations
                for _ in range(rem):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()
                break

            # Sample population
            ys = np.zeros((lam, n))
            fs = np.zeros(lam)
            gen_best_fx = fs[0]
            for i in range(lam):
                z = rng.standard_normal(n)
                ys[i] = x_mean + sigma * (C_inv_sqrt @ z)
                ys[i] = np.clip(ys[i], -5, 5)
                fs[i] = func(ys[i])
                calls_used += 1
                if fs[i] < gen_best_fx:
                    gen_best_fx = fs[i]
                if fs[i] < best_fx:
                    best_fx = fs[i]
                    best_x = ys[i].copy()

            # Check for improvement this generation
            if gen_best_fx < best_fx + 1e-10:
                # There was some improvement potential
                pass
            # Check if overall best improved
            if best_fx < self._prev_best if hasattr(self, '_prev_best') else True:
                no_improve_count = 0
            else:
                no_improve_count += 1

            self._prev_best = best_fx

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]

            # Update mean
            x_mean_new = np.sum(w[:, None] * ys_sorted[:mu], axis=0)

            # Update evolution paths
            y_vec = (x_mean_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec

            # Update covariance
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            rank_mu = np.outer(pc, pc) + delta_h * np.outer(pc, pc)
            rank_1 = np.outer(y_vec, y_vec)
            C = (1 + c_sigma * (c_cov_pos - c_sigma) + c_cov_neg * (mu_eff - 2) - c_sigma * c_cov_neg) * C \
                + c_cov_pos * rank_1 + c_cov_neg * (np.sum(w) - 1) * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 100.0)

            # Update C^{-1/2}
            D, B = np.linalg.eigh(C)
            D = np.maximum(D, 1e-12)
            C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

            x_mean = x_mean_new

            # Stagnation handling: if no improvement for many generations,
            # inject diversity by resetting mean toward best and increasing sigma
            if no_improve_count >= stagnation_threshold and sigma > 0.05:
                # Restart from best point with larger sigma
                x_mean = best_x.copy()
                sigma *= 2.0
                sigma = np.clip(sigma, 0.1, 50.0)
                C = np.eye(n)
                D, B = np.linalg.eigh(C)
                D = np.maximum(D, 1e-12)
                C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T
                pc = np.zeros(n)
                ps = np.zeros(n)
                no_improve_count = 0

        # Final polish: use remaining budget for local refinement around best_x
        remaining = budget - calls_used
        if remaining > 2 and time.time() < deadline - 5:
            # Simple local search: try small perturbations along principal axes of C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Sort by eigenvalue descending (most important directions first)
            order = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, order]
            
            polish_sigma = max(sigma * 0.1, 1e-4)
            for _ in range(min(remaining, 20)):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break
                # Random direction in the covariance space
                z = rng.standard_normal(n)
                y = best_x + polish_sigma * (eigvecs @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    # Shrink polish sigma on improvement
                    polish_sigma *= 0.8
                else:
                    polish_sigma *= 0.95

        return best_x, best_fx
# EVOLVE-BLOCK-END
