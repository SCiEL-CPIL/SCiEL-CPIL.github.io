"""Combined optimizer: CMA-ES global search + Nelder-Mead local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for global basin location followed by Nelder-Mead for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before the 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: CMA-ES for global search ---
        # Population size
        lambda_pop = int(4 + 3 * np.log(dim))
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x_mean = rng.uniform(-2, 2, dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # CMA-ES main loop
        cma_budget = int(budget * 0.6)
        max_cma_iters = (cma_budget - call_count) // lambda_pop
        max_cma_iters = max(1, min(max_cma_iters, 50))

        for it in range(max_cma_iters):
            if time.time() > deadline or call_count >= budget:
                break

            # Generate population
            population = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                x_new = x_mean + sigma * (B @ (D * z))
                x_new = np.clip(x_new, -5.0, 5.0)
                population.append(x_new)

            # Evaluate population
            f_values = [call(x) for x in population]
            if call_count >= budget:
                break

            # Sort by fitness
            idx = np.argsort(f_values)
            population = [population[i] for i in idx]
            f_values = [f_values[i] for i in idx]

            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = sum(weights[i] * population[i] for i in range(mu))
            x_mean = np.clip(x_mean, -5.0, 5.0)

            # Update covariance matrix
            if sigma > 1e-12:
                y = (x_mean_old - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (it + 1)) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0

                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
                rank_one = np.outer(p_c, p_c)
                rank_mu = sum(weights[i] * np.outer(population[i] - x_mean_old, population[i] - x_mean_old) for i in range(mu)) / sigma ** 2

                C = (1 + c_1 * h_sigma * (c_c + 2 - c_c) - c_1 - c_mu) * C + c_1 * (rank_one - C) + c_mu * rank_mu

                # Recompute B and D via eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                B = eigvecs
                D = np.sqrt(eigvals)

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

        # --- Phase 2: Nelder-Mead for local refinement ---
        if call_count < budget and best_x is not None:
            x_center = best_x.copy()
            step_size = 0.5
            nm_budget = budget - call_count
            nm_max_iters = max(10, nm_budget // (dim + 2))

            # Create initial simplex
            simplex = [x_center.copy()]
            for i in range(dim):
                p = x_center.copy()
                p[i] += step_size
                simplex.append(np.clip(p, -5.0, 5.0))

            f_values = [call(p) for p in simplex]
            if call_count >= budget:
                pass
            else:
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_nm = 0.5

                for _ in range(nm_max_iters):
                    if time.time() > deadline or call_count >= budget:
                        break

                    # Sort simplex
                    sorted_idx = np.argsort(f_values)
                    simplex = [simplex[i] for i in sorted_idx]
                    f_values = [f_values[i] for i in sorted_idx]

                    # Check convergence
                    if abs(f_values[0] - f_values[-1]) < 1e-10:
                        break

                    # Centroid of best N points
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    x_ref = centroid + alpha * (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, -5.0, 5.0)
                    f_ref = call(x_ref)

                    if f_ref < f_values[0]:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5.0, 5.0)
                        f_exp = call(x_exp)
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            f_values[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            f_values[-1] = f_ref
                    elif f_ref < f_values[-2]:
                        # Accept reflection
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                    else:
                        # Contraction
                        if f_ref < f_values[-1]:
                            x_con = centroid + rho * (x_ref - centroid)
                        else:
                            x_con = centroid - rho * (centroid - simplex[-1])
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = call(x_con)

                        if f_con < min(f_ref, f_values[-1]):
                            simplex[-1] = x_con
                            f_values[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_values[i] = call(simplex[i])
                                if call_count >= budget:
                                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
