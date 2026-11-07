"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with aggressive restarts for multimodal BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls = 0

        # CMA-ES parameters
        lambda_pop = 10
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

            # Pre-compute eigendecomposition of C for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_eigvals = np.sqrt(eigvals)
            L = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
            
            # Helper to evaluate a population
            def eval_pop():
                nonlocal calls
                zs = rng.standard_normal((lambda_pop, n))
                # y = x_mean + sigma * (L @ z.T).T
                # Efficiently: y = x_mean[None, :] + sigma * (L @ zs.T).T
                pop_x = x_mean[None, :] + sigma * (L @ zs.T).T
                pop_x = np.clip(pop_x, -5, 5)
                pop_f = np.array([func(x) for x in pop_x])
                calls += lambda_pop
                indices = np.argsort(pop_f)
                return pop_x[indices], pop_f[indices]

            if calls + lambda_pop > max_evals:
                return x0, func(x0)

            pop_x, pop_f = eval_pop()
            best_x = pop_x[0].copy()
            best_f = pop_f[0]
            evals_used = lambda_pop
            no_improve = 0
            
            # Thresholds for restart
            stagnation_limit = 15
            sigma_min = 1e-8
            sigma_max = 1e10

            while evals_used < max_evals:
                # Update mean
                x_parents = pop_x[:mu]
                x_mean_new = np.sum(w[:, None] * x_parents, axis=0)

                # Compute C^{-1/2} using current eigendecomposition
                # Note: We need to update eigendecomp of C after C is updated, 
                # but for the update step we use the OLD C's inverse sqrt.
                inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
                inv_L = eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T

                # y_i = C^{-1/2} (x_i - x_mean) / sigma
                y_parents = inv_L @ ((x_parents - x_mean).T / sigma)

                # Update ps
                # The term (1 - c_sigma) ** (2 * t) approximates the survival probability
                # For simplicity and speed, we use the standard check
                hsig = (
                    np.linalg.norm(ps)
                    / np.sqrt(1 - (1 - c_sigma) ** (2 * (max_evals - evals_used + lambda_pop)))
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
                # C = (1 - c_1 - c_mu) * C + c_1 * (n + 2 - c_1) * C? No.
                # Standard update:
                # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - hsig) * c_c * (2 - c_c)) + c_mu * sum(w_i * y_i * y_i.T)
                # Wait, the standard formula is:
                # C = (1 - c_1 - c_mu) * C + c_1 * (n + 2 - c_1) * C is WRONG.
                # It is: C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hsig) * c_c * (2 - c_c)) + c_mu * ...
                # Actually, the term is c_1 * (pc @ pc.T) + c_1 * (1 - hsig) * c_c * (2 - c_c) is not quite right either.
                # Let's use the robust formulation:
                # C = (1 - c_1 - c_mu) * C
                #   + c_1 * (pc @ pc.T + (1 - hsig) * c_c * (2 - c_c))
                #   + c_mu * sum(w_i * y_i @ y_i.T)
                
                # Correction: The term for pc is usually added as part of the rank-1 update.
                # The standard CMA-ES update is:
                # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hsig) * c_c * (2 - c_c)) + c_mu * H
                # where H = sum(w_i * y_i @ y_i.T)
                
                H = np.sum(w[:, None, None] * y_parents[:, :, None] * y_parents[:, None, :], axis=1)
                
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + delta_hsig) + c_mu * H
                
                # Ensure symmetry and PSD
                C = (C + C.T) / 2
                eigvals_C, eigvecs_C = np.linalg.eigh(C)
                eigvals_C = np.maximum(eigvals_C, 1e-12)
                C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T
                
                # Update eigendecomposition for next sampling
                eigvals, eigvecs = eigvals_C, eigvecs_C
                sqrt_eigvals = np.sqrt(eigvals)
                L = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T

                # Update sigma
                sigma *= np.exp(
                    c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1)
                )
                sigma = np.clip(sigma, sigma_min, sigma_max)

                x_mean = x_mean_new

                # Evaluate new population
                pop_x, pop_f = eval_pop()
                evals_used += lambda_pop

                if pop_f[0] < best_f - 1e-12:
                    best_f = pop_f[0]
                    best_x = pop_x[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Restart condition
                if no_improve >= stagnation_limit or sigma < sigma_min:
                    break

            return best_x, best_f

        # Multi-start CMA-ES with aggressive restarts
        best_x_overall = None
        best_f_overall = np.inf

        # Strategy: 
        # 1. Try a few random starts with small budgets to find promising regions.
        # 2. If a good region is found, refine it.
        # 3. If not, try a larger start.
        
        # Phase 1: Exploration
        # We allocate 50% of budget to exploration, 50% to exploitation/refinement.
        # But we do it dynamically.
        
        starts = [
            (rng.uniform(-4, 4, n), 2.0),
            (rng.uniform(-4, 4, n), 1.5),
            (rng.uniform(-4, 4, n), 1.0),
            (np.zeros(n), 1.0)
        ]
        
        # Give each start a portion of the budget, but allow early stopping if converged
        budget_per_start = total_budget // len(starts)
        
        for x0, sigma0 in starts:
            if calls >= total_budget:
                break
            
            remaining = total_budget - calls
            if remaining < lambda_pop:
                break
            
            # Limit each start to at most 1/4 of total budget initially
            max_evals = min(budget_per_start, remaining)
            
            x_best, f_best = cma_es(x0, sigma0, max_evals)
            
            if f_best < best_f_overall:
                best_f_overall = f_best
                best_x_overall = x_best

        # Phase 2: Refinement around the best found point
        if calls < total_budget and best_x_overall is not None:
            remaining = total_budget - calls
            if remaining >= lambda_pop:
                # Refine with smaller sigma
                x_best, f_best = cma_es(best_x_overall.copy(), 0.5, remaining)
                if f_best < best_f_overall:
                    best_f_overall = f_best
                    best_x_overall = x_best
        
        if best_x_overall is None:
            best_x_overall = np.zeros(n)
            best_f_overall = func(best_x_overall)
            calls += 1

        return best_x_overall, best_f_overall
# EVOLVE-BLOCK-END
