"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with efficient sampling and aggressive restarts for multimodal BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls = 0

        # CMA-ES parameters
        # Standard lambda is 4 + floor(3 log n). For n=5, log(5)~1.6, 3*1.6=4.8, floor=4, lambda=8.
        # For multimodal, we might want slightly larger. Let's use 14 (2*n+4) or 10.
        # Let's stick to standard guidelines but ensure it's not too small.
        lambda_pop = 14 
        mu = lambda_pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 + 2 * min(w[0], 0.5) * (mu_eff - 1) / (n + 2)) / (
            (n + 1.3) ** 2 + mu_eff
        )
        c_mu = min(
            1 - c_1,
            2 * (
                (sqrt_mu_eff := np.sqrt(mu_eff)) * (w[0] + w[1] - 2) / (n + 2)
            ),
        )
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        def cma_es(x0, sigma0, max_evals):
            nonlocal calls
            x_mean = x0.astype(float).copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Pre-compute Cholesky for initial identity
            L = np.eye(n)

            def sample_and_eval():
                nonlocal calls
                pop = []
                for _ in range(lambda_pop):
                    z = rng.standard_normal(n)
                    # Use L for sampling: y = x_mean + sigma * L @ z
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    pop.append((y, fy))
                pop.sort(key=lambda t: t[1])
                return pop

            if calls + lambda_pop > max_evals:
                return x0, func(x0)

            pop = sample_and_eval()
            best_x = pop[0][0].copy()
            best_f = pop[0][1]

            evals_used = lambda_pop
            no_improve = 0
            best_f_so_far = best_f

            while evals_used < max_evals:
                # Update mean
                x_parents = np.array([pop[i][0] for i in range(mu)])
                x_mean_new = np.sum(w[:, None] * x_parents, axis=0)

                # Compute C^{-1/2} for updating ps and pc
                # We need to transform (x_mean_new - x_mean) / sigma into the normalized space
                # This requires C^{-1/2}. We can use eigendecomposition once here.
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
                inv_L = eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T

                # y_i = C^{-1/2} (x_i - x_mean) / sigma
                y_parents = inv_L @ ((x_parents - x_mean).T / sigma)  # (n, mu)

                # Update ps
                # hsig condition: norm(ps) < (1 + 2/(n+1)) * chi_n * sqrt(1 - (1-c_sigma)^(2*t))
                # Approximate t as remaining budget / lambda, or just use current step count?
                # Standard CMA-ES uses generation count.
                gen_count = evals_used // lambda_pop
                hsig = (
                    np.linalg.norm(ps)
                    / np.sqrt(1 - (1 - c_sigma) ** (2 * gen_count))
                    < (1 + 2 / (n + 1)) * chi_n
                )
                
                ps = (1 - c_sigma) * ps + hsig * np.sqrt(
                    c_sigma * (2 - c_sigma) * mu_eff
                ) * inv_L @ ((x_mean_new - x_mean) / sigma)

                # Update pc
                pc = (
                    1 - c_c
                ) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma

                # Update C
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (
                    (1 - c_1 - c_mu) * C
                    + c_1 * (n + 2 - c_1) * C
                    - c_1 * pc[:, None] @ pc[None, :]
                    + c_mu * np.sum(
                        w[:, None, None] * y_parents[:, :, None] * y_parents[:, None, :], axis=1
                    )
                    - c_1 * delta_hsig * pc[:, None] @ pc[None, :]
                )
                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2
                # Project to PSD and compute new Cholesky
                eigvals_C, eigvecs_C = np.linalg.eigh(C)
                eigvals_C = np.maximum(eigvals_C, 1e-12)
                C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T
                
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to eigendecomposition if Cholesky fails
                    eigvals_C, eigvecs_C = np.linalg.eigh(C)
                    eigvals_C = np.maximum(eigvals_C, 1e-12)
                    L = eigvecs_C @ np.diag(np.sqrt(eigvals_C))

                # Update sigma
                sigma *= np.exp(
                    c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1)
                )
                sigma = np.clip(sigma, 1e-10, 1e10)

                x_mean = x_mean_new

                # Evaluate new population
                pop = sample_and_eval()
                evals_used += lambda_pop

                if pop[0][1] < best_f_so_far - 1e-12:
                    best_f_so_far = pop[0][1]
                    best_x = pop[0][0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Restart condition: No improvement for 20 generations
                if no_improve >= 20:
                    break

            return best_x, best_f_so_far

        # Main loop: Multi-start with restarts
        best_x_overall = None
        best_f_overall = np.inf
        
        # Initial broad exploration
        # We will run CMA-ES instances. If one stagnates, we start a new one.
        # We keep the global best and use it to seed new runs if needed, or just random.
        
        remaining = total_budget
        run_id = 0
        current_sigma = 2.0
        
        while remaining >= lambda_pop:
            run_id += 1
            # Determine starting point
            if best_x_overall is None:
                x0 = rng.uniform(-4, 4, n)
            else:
                # With some probability, restart from global best, otherwise random
                # This helps escape local optima while refining the best found
                if rng.random() < 0.3:
                    x0 = best_x_overall + rng.normal(0, 0.5, n)
                    x0 = np.clip(x0, -4.5, 4.5)
                else:
                    x0 = rng.uniform(-4, 4, n)
            
            # Determine sigma for this run
            # Start large, decrease if we restart from best?
            # For now, use a fixed moderate sigma for exploration, 
            # and let CMA-ES adapt.
            sigma0 = current_sigma
            
            # Allocate budget for this run
            # Give it a chunk of budget, e.g., 100 evals or until stagnation
            # If it stagnates early, we restart.
            max_evals_for_run = min(100, remaining)
            
            x_best, f_best = cma_es(x0, sigma0, max_evals_for_run)
            
            if f_best < best_f_overall:
                best_f_overall = f_best
                best_x_overall = x_best
            
            remaining = total_budget - calls
            
            # If we used all budget, break
            if remaining < lambda_pop:
                break

            # If we stagnated (no_improve triggered break), we might want to increase sigma for next run?
            # Or just keep it. The random restarts handle diversity.
            
        if best_x_overall is None:
            best_x_overall = np.zeros(n)
            best_f_overall = func(best_x_overall)
            calls += 1

        return best_x_overall, best_f_overall
# EVOLVE-BLOCK-END
