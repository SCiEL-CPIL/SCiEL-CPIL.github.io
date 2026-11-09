"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts and global exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(12345)

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf
        eval_count = 0

        def check_time():
            return time.time() < deadline - 5.0

        def record(x, fx):
            nonlocal global_best_x, global_best_f, eval_count
            eval_count += 1
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

        def cma_es(start_x, max_evals, sigma_init=1.0, rng_seed=None):
            """Run CMA-ES from start_x for up to max_evals evaluations."""
            nonlocal global_best_x, global_best_f, eval_count, deadline
            if rng_seed is not None:
                local_rng = np.random.default_rng(rng_seed)
            else:
                local_rng = np.random.default_rng()

            x = start_x.copy().astype(float)
            fx = func(x)
            record(x, fx)
            if eval_count >= max_evals or not check_time():
                return x, fx

            # CMA-ES parameters
            pop_size = max(4, int(np.ceil(4 + 3 * np.log(n))))
            sigma = sigma_init
            weights = np.exp(-2 * np.arange(1, pop_size + 1) / pop_size)
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()

            # Learning rates
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # State
            mean = x.copy()
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            best_local_x = x.copy()
            best_local_f = fx

            while eval_count < max_evals and check_time():
                # Generate population
                D = np.linalg.cholesky(C)
                B = np.linalg.inv(D).T  # So that B @ D @ D.T @ B.T = I
                pop = np.zeros((pop_size, n))
                pop_f = np.zeros(pop_size)

                for i in range(pop_size):
                    z = local_rng.standard_normal(n)
                    y = mean + sigma * (D @ z)
                    y = np.clip(y, -5, 5)
                    pop[i] = y
                    pop_f[i] = func(y)
                    record(y, pop_f[i])

                    if eval_count >= max_evals or not check_time():
                        break

                if eval_count >= max_evals or not check_time():
                    break

                # Sort by fitness
                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]

                # Update best local
                if pop_f[0] < best_local_f:
                    best_local_f = pop_f[0]
                    best_local_x = pop[0].copy()

                # Compute weighted mean
                m_new = np.average(pop[:int(mu_eff) + 1], axis=0, weights=weights[:int(mu_eff) + 1])

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / np.linalg.norm(D) * (B @ (m_new - mean) / sigma)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (max_evals - eval_count)))) < (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (m_new - mean) / sigma

                # Update covariance
                delta_hsig = (1.0 if hsig else 0.0) - (1 - cc)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C
                C += c1 * (np.outer(pc, pc) - C)
                C += cmu / mu_eff * np.sum([np.outer(pop[i] - mean, pop[i] - mean) for i in range(int(mu_eff) + 1)], axis=0) * weights[:int(mu_eff) + 1].reshape(-1, 1, 1)

                # Ensure positive definiteness
                try:
                    _ = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-6 * np.eye(n)

                # Update sigma (CSA)
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                # Update mean
                mean = m_new
                mean = np.clip(mean, -5, 5)

                # Stagnation check
                if np.allclose(pop_f[:5], pop_f[0], atol=1e-10) and sigma < 1e-5:
                    break

            return best_local_x, best_local_f

        # === Phase 1: Multi-start with diverse initializations ===
        # Start 1: Origin
        if check_time() and eval_count < budget:
            cma_es(np.zeros(n), budget, sigma_init=1.5, rng_seed=100)

        # Start 2: Random point in [-3,3]
        if check_time() and eval_count < budget:
            x2 = rng.uniform(-3, 3, n)
            cma_es(x2, budget, sigma_init=1.0, rng_seed=200)

        # Start 3: Another random point
        if check_time() and eval_count < budget:
            x3 = rng.uniform(-4, 4, n)
            cma_es(x3, budget, sigma_init=0.8, rng_seed=300)

        # Start 4: Perturbation of global best
        if check_time() and eval_count < budget and global_best_x is not None:
            x4 = np.clip(global_best_x + rng.standard_normal(n) * 0.5, -5, 5)
            cma_es(x4, budget, sigma_init=0.5, rng_seed=400)

        # Start 5: One more random if budget allows
        if check_time() and eval_count < budget:
            x5 = rng.uniform(-2, 2, n)
            cma_es(x5, budget, sigma_init=0.7, rng_seed=500)

        # === Phase 2: Final local refinement around global best ===
        if check_time() and eval_count < budget and global_best_x is not None:
            x_ref = global_best_x.copy()
            f_ref = global_best_f
            sigma = 0.2
            no_improve = 0
            while eval_count < budget and check_time() and no_improve < 50:
                z = rng.standard_normal(n)
                y = np.clip(x_ref + sigma * z, -5, 5)
                fy = func(y)
                record(y, fy)

                if fy < f_ref:
                    x_ref = y
                    f_ref = fy
                    sigma = min(sigma * 1.3, 5.0)
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
