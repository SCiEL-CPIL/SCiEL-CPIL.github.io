# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """CMA-ES with structured initialization and adaptive restarts for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        evals_used = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def evaluate(x):
            nonlocal best_f, best_x, evals_used
            x = clip(x)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cma_es_full(x0, sigma0, max_evals, lambda_n=None):
            """Full CMA-ES with lambda offspring per generation."""
            nonlocal best_f, best_x, evals_used
            n = dim
            if lambda_n is None:
                lambda_n = int(4 + 3 * n)

            mu = lambda_n // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_sqrt_inv = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            c_sigma = 1.0 / (n + 3)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            d_sigma = 1 + 2 * max(0, 1 / np.sqrt(n + 1) - 1) + c_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            def recompute_csqrt():
                nonlocal C, C_inv, C_sqrt, C_sqrt_inv
                C += 1e-10 * np.eye(n)
                C_inv = np.linalg.inv(C)
                C_sqrt = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(C_sqrt)

            recompute_csqrt()
            gen = 0
            no_improve = 0
            best_local = best_f

            while evals_used < max_evals and time.time() < deadline:
                if sigma < 1e-12:
                    break

                # Generate offspring
                offspring = []
                f_offspring = []
                for i in range(lambda_n):
                    if time.time() > deadline or evals_used >= budget:
                        return
                    z = rng.standard_normal(n)
                    y = C_sqrt @ z
                    x_trial = x + sigma * y
                    f_trial = evaluate(x_trial)
                    offspring.append(x_trial)
                    f_offspring.append(f_trial)

                gen += 1

                # Sort offspring
                order = np.argsort(f_offspring)
                offspring = np.array(offspring)[order]
                f_offspring = np.array(f_offspring)[order]

                # Update best
                if f_offspring[0] < best_local:
                    best_local = f_offspring[0]
                    no_improve = 0
                else:
                    no_improve += 1

                # Recompute C_sqrt every 5 generations for numerical stability
                if gen % 5 == 0:
                    recompute_csqrt()

                # Update p_sigma
                y_sigma = (C_sqrt_inv @ (offspring[0] - x) / sigma)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                    c_sigma * (2 - c_sigma) * mueff) * y_sigma
                norm_ps = np.linalg.norm(p_sigma)
                h_sigma = norm_ps / np.sqrt(
                    1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n
                if h_sigma:
                    sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

                # Update p_c
                y_c = (C_inv @ (offspring[0] - x) / sigma)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_c
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)

                # Update C
                y_w = np.array([(C_inv @ (offspring[i] - x) / sigma) for i in range(mu)])
                C = ((1 + c_1 * (delta_h - 1) - c_1 - c_mu) * C +
                     c_1 * np.outer(p_c, p_c) +
                     c_mu * np.dot(w, y_w) @ y_w.T)

                # Update mean
                x = np.sum(w[:, None] * offspring[:mu], axis=0)

                # Stagnation check
                if no_improve > 30 and sigma < 1e-6:
                    break

        def nelder_mead(x0, max_evals, step=0.3):
            """Simple Nelder-Mead for local refinement."""
            nonlocal best_f, best_x, evals_used
            n = dim
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0.copy()
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += step

            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if time.time() > deadline or evals_used >= budget:
                    return
                fvals[i] = evaluate(simplex[i])

            alpha, gamma, rho, sigma_sh = 1.0, 2.0, 0.5, 0.5

            for _ in range(max_evals):
                if time.time() > deadline or evals_used >= budget:
                    return

                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                centroid = np.mean(simplex[:-1], axis=0)
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = evaluate(x_r)

                if f_r < fvals[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = evaluate(x_c)
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_sh * (simplex[i] - simplex[0])
                            if time.time() > deadline or evals_used >= budget:
                                return
                            fvals[i] = evaluate(simplex[i])
                        break

        # === Phase 1: Structured diverse initialization ===
        # Center
        x_center = np.zeros(dim)
        evaluate(x_center)

        # Midpoints of edges from center
        for i in range(dim):
            for sign in [-1, 1]:
                if time.time() > deadline or evals_used >= budget:
                    break
                xm = np.zeros(dim)
                xm[i] = sign * 3.0
                evaluate(xm)

        # Random points in [-4, 4]^dim
        n_random = 30
        for i in range(n_random):
            if time.time() > deadline or evals_used >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            evaluate(x)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # === Phase 2: Full CMA-ES from best point ===
        remaining = budget - evals_used
        cma_budget1 = min(200, remaining * 2 // 3)
        cma_es_full(best_x.copy(), sigma0=1.0, max_evals=cma_budget1)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # === Phase 3: Nelder-Mead local refinement ===
        remaining = budget - evals_used
        nm_budget = min(50, remaining // 2)
        nelder_mead(best_x.copy(), nm_budget, step=0.2)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # === Phase 4: Restart CMA-ES from best with smaller sigma ===
        remaining = budget - evals_used
        cma_budget2 = min(150, remaining * 2 // 3)
        cma_es_full(best_x.copy(), sigma0=0.3, max_evals=cma_budget2)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # === Phase 5: Additional Nelder-Mead ===
        remaining = budget - evals_used
        nm_budget2 = min(40, remaining)
        nelder_mead(best_x.copy(), nm_budget2, step=0.1)

        if time.time() > deadline or evals_used >= budget:
            return best_x, best_f

        # === Phase 6: Final CMA-ES polish ===
        remaining = budget - evals_used
        if remaining > 30:
            cma_es_full(best_x.copy(), sigma0=0.1, max_evals=remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
