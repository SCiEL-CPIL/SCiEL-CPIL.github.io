"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Full CMA-ES with multi-restart for Rastrigin rotated (f15)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        box_lo, box_hi = -5.0, 5.0

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_f, best_x
            x = np.clip(x, box_lo, box_hi)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Initial exploration (spread out for early AOCC progress) ---
        n_init = 25
        # Mix of center + spread + boundary
        for i in range(n_init):
            if i == 0:
                x = np.zeros(dim)
            elif i < 5:
                # Centered small perturbations
                x = 0.5 * rng.standard_normal(dim)
            elif i < 15:
                # Medium spread
                x = 2.0 * rng.standard_normal(dim)
            else:
                # Wide spread
                x = rng.uniform(-4.5, 4.5, dim)
            evaluate(x)

        if best_x is None:
            return (np.zeros(dim), 0.0)

        # --- Phase 2: CMA-ES with multi-restart ---
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # ~8 for dim=5
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))

        def cma_es(x0, sigma0, max_calls, time_limit):
            nonlocal calls_used, best_f, best_x
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim ** 2))

            # Evaluate initial population
            pop = []
            for j in range(lambda_):
                if calls_used >= budget or time.time() > time_limit:
                    break
                z = rng.standard_normal(dim)
                y = np.clip(x_mean + sigma * z, box_lo, box_hi)
                fy = evaluate(y)
                pop.append((fy, y, z))

            if not pop:
                return

            # Sort
            pop.sort(key=lambda t: t[0])

            while calls_used < budget and time.time() < time_limit:
                # Update best from this generation
                # (already done in evaluate)

                # Compute weighted mean
                y_mean = np.zeros(dim)
                for i in range(min(mu, len(pop))):
                    y_mean += weights[i] * pop[i][1]
                y_mean /= weights[:min(mu, len(pop))].sum()

                # Update CMA state
                C_inv_sqrt_approx = np.linalg.inv(np.sqrtm(C + 1e-10 * np.eye(dim)))
                diff = (y_mean - x_mean) / sigma
                pc = (1 - c_c) * pc + c_c * diff
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt_approx @ diff

                # Check ps norm for step size adaptation
                ps_norm = np.linalg.norm(ps)
                sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)

                # Update covariance
                delta_heavy = (1 - c_c) if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + c_c) * chi_n else 0
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(min(mu, len(pop))):
                    z_i = (pop[i][1] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(z_i, z_i)
                C = (1 + c_c * delta_heavy - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                x_mean = y_mean.copy()

                # Generate next population
                pop = []
                for j in range(lambda_):
                    if calls_used >= budget or time.time() > time_limit:
                        break
                    z = rng.standard_normal(dim)
                    # Transform z by C^(1/2)
                    z_transformed = eigvecs @ (np.sqrt(eigvals) * z)
                    y = np.clip(x_mean + sigma * z_transformed, box_lo, box_hi)
                    fy = evaluate(y)
                    pop.append((fy, y, z))

                if not pop:
                    break
                pop.sort(key=lambda t: t[0])

                # Stagnation check: if sigma is very small, we're converged
                if sigma < 1e-4:
                    break

            return

        # Multi-restart CMA-ES
        # First CMA-ES from best found
        remaining = budget - calls_used
        if remaining > 20:
            cma_es(best_x.copy(), 1.0, budget, deadline)

        # Periodicity-aware restarts: Rastrigin has period 2
        # Try restarts at best_x +/- 2 in each direction (in rotated space approx)
        n_restarts = 4
        restart_budget_each = max(15, (budget - calls_used) // (n_restarts + 1))

        for r in range(n_restarts):
            if calls_used >= budget - 5 or time.time() > deadline:
                break
            # Periodicity-aware: shift by 2 (Rastrigin period)
            # Also try random restarts
            if r < 2:
                # Shift by ±2 in random directions
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                x_restart = best_x + 2.0 * direction
                x_restart = np.clip(x_restart, -4.5, 4.5)
            else:
                # Random restart
                x_restart = rng.uniform(-4.5, 4.5, dim)

            fx_restart = evaluate(x_restart)
            if calls_used >= budget - 10 or time.time() > deadline:
                break

            # Short CMA-ES from this point
            cma_es(x_restart.copy(), 1.5, budget, deadline)

        # --- Phase 3: Final polish (tight local search) ---
        if calls_used < budget - 10 and time.time() < deadline:
            # Local Nelder-Mead style polish
            x_polish = best_x.copy()
            f_polish = best_f
            step = 0.1
            for _ in range(20):
                if calls_used >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        if calls_used >= budget or time.time() > deadline:
                            break
                        x_try = x_polish.copy()
                        x_try[d] += sign * step
                        x_try = np.clip(x_try, box_lo, box_hi)
                        f_try = evaluate(x_try)
                        if f_try < f_polish - 1e-10:
                            x_polish = x_try
                            f_polish = f_try
                            improved = True
                if improved:
                    step *= 0.8
                else:
                    step *= 0.5
                if step < 1e-5:
                    break

        return (best_x, best_f)
# EVOLVE-BLOCK-END
