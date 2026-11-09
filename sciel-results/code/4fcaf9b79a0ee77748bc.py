"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with time guard for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        n = self.dim
        total_budget = self.budget

        best_x = np.zeros(n)
        best_f = np.inf

        # Multi-restart strategy:
        # Phase 1: Global exploration with larger sigma (200 evals)
        # Phase 2: Local refinement around best found (200 evals)
        # Phase 3: Final polish (100 evals)

        phases = [
            {"budget": 200, "sigma_init": 3.0, "pop_size": 10},
            {"budget": 200, "sigma_init": 1.5, "pop_size": 8},
            {"budget": 100, "sigma_init": 0.5, "pop_size": 6},
        ]

        for phase_idx, phase in enumerate(phases):
            if time.time() > deadline:
                break

            phase_budget = phase["budget"]
            sigma = phase["sigma_init"]
            pop_size = phase["pop_size"]

            # Initialize center
            if phase_idx == 0:
                rng = np.random.default_rng(0)
                mean = rng.uniform(-5, 5, n)
            else:
                mean = best_x.copy()

            # CMA-ES state
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Evaluation counter for this phase
            evals_used = 0

            # Initial evaluation
            mean_clipped = np.clip(mean, -5, 5)
            f_mean = func(mean_clipped)
            evals_used += 1
            if f_mean < best_f:
                best_f = f_mean
                best_x = mean_clipped.copy()

            # CMA-ES parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)
            c1 = (2 + 2 / (n + 1) ** 2) / ((n + 2) ** 2 + 2)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / n) / ((n + 2) ** 2 + mueff))
            cs = (mueff + 2 + 2 / (n + 1)) / (n + mueff + 5)
            ds = 1 + cs + 2 * max(0, np.sqrt((cs - 1) * (mueff + 1) / (n + 1)) - 1)

            # Main loop
            while evals_used < phase_budget:
                if time.time() > deadline:
                    break

                # Generate population
                rng = np.random.default_rng(hash((phase_idx, evals_used)) % (2**32))
                z = rng.standard_normal((pop_size, n))
                
                # Cholesky decomposition for C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    L = np.eye(n)

                pops = mean[None, :] + sigma * (z @ L.T)
                pops = np.clip(pops, -5, 5)

                # Evaluate all
                fvals = np.array([func(pops[i]) for i in range(pop_size)])
                evals_used += pop_size

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = pops[min_idx].copy()

                # Sort
                order = np.argsort(fvals)
                sorted_pops = pops[order]
                sorted_fvals = fvals[order]

                # Weighted recombination
                new_mean = np.sum(weights[:, None] * sorted_pops[:mu], axis=0)

                # Update covariance matrix
                y = (new_mean - mean) / sigma
                pc = (1 - cs) * pc + cs * (np.sqrt(ds * cs * (2 - cs)) / chi) * y
                delta_sig = (np.linalg.norm(ps) / chi - 1) > (1.4 + 2 / (n + 1))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * delta_sig) * y
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (sorted_pops[i] - mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)
                C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

                # Update mean and sigma
                mean = new_mean
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
