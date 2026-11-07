"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size adapted to budget, plus multi-start restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        n = self.dim
        budget = self.budget

        # Track best across all runs
        global_best_x = None
        global_best_f = np.inf

        # CMA-ES parameters
        # Use a reasonable population size
        lam = max(4, int(4 + 3 * np.log(n)))  # population size
        max_iter = max(1, (budget - 1) // lam)  # number of generations

        # Multi-start: try a few different initializations
        n_starts = min(3, budget // 100)  # at most 3 restarts, each needs ~100 evals
        evals_used = 0

        for start_idx in range(n_starts):
            if evals_used >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize
            if start_idx == 0:
                x = np.zeros(n)
            else:
                # Random restart
                x = np.random.uniform(-4, 4, n)

            fx = func(x)
            evals_used += 1

            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

            # CMA-ES state
            sigma = 2.0  # initial step size
            C = np.eye(n)  # covariance matrix
            pc = np.zeros(n)  # evolution path for C
            ps = np.zeros(n)  # evolution path for sigma
            c_sigma = 1.0 / (lam + 3)
            d_sigma = 1.0 + 2 * max(0, (np.sqrt((lam + 1 - 3) / (n + 1)) - 1)) / (lam + 3)
            c_c = (4 + lam / n) / (n + 4)
            mu = lam // 2  # number of recombination weights
            # Weights (logarithmic)
            weights = np.log(np.arange(mu, 0, -1)) - np.log(mu)
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_c = (4 + mu_eff / n) / (n + 4)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)

            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # Generate initial population
            for it in range(max_iter):
                if evals_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                zs = np.random.randn(lam, n)
                # Apply Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                ys = np.array([zs[i] @ L for i in range(lam)])
                xs = np.array([x + sigma * ys[i] for i in range(lam)])
                # Clip to box
                xs = np.clip(xs, -5, 5)

                # Evaluate
                fs = np.array([func(xs[i]) for i in range(lam)])
                evals_used += lam

                # Sort by fitness
                order = np.argsort(fs)
                xs_sorted = xs[order]
                fs_sorted = fs[order]

                # Update best
                if fs_sorted[0] < global_best_f:
                    global_best_f = fs_sorted[0]
                    global_best_x = xs_sorted[0].copy()

                # Weighted recombination
                m_new = np.sum([weights[i] * xs_sorted[i] for i in range(mu)], axis=0)

                # Update sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (m_new - x)
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (it + 1)) < (1.4 + 1 / n) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update C
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - x) / sigma
                delta_h = 1 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (it + 1)) < (1.4 + 1 / n) * chi_n else 0
                rank_mu = np.outer(m_new - x, m_new - x) / sigma ** 2
                rank_one = np.outer(pc, pc)
                C = (1 + c_c * (2 - c_c) - c_sigma) * C + c_sigma * (rank_mu + (1 - delta_h + c_sigma) * rank_one)

                # Update mean
                x = m_new

                # Periodic re-clip
                x = np.clip(x, -5, 5)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
