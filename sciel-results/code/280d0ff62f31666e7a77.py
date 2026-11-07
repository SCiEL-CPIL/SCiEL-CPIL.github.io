"""CMA-ES optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 10 and fixed covariance adaptation for f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls_made < budget - 1

        # CMA-ES parameters
        lambda_ = 10  # population size
        mu = lambda_ // 2  # number of parents

        # Weights (log-sum normalization)
        weights = np.log(np.arange(mu, 0, -1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size initialization
        sigma0 = 2.0

        # Number of restarts
        max_restarts = 4

        # CMA-ES rate parameters (fixed)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 1 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        for restart in range(max_restarts):
            if not (budget_ok() and time_ok()):
                break

            # Initial center
            if restart == 0:
                center = np.zeros(dim)
            else:
                rng_init = np.random.default_rng(2000 + restart * 13)
                center = rng_init.uniform(-4.0, 4.0, dim)

            # CMA-ES state
            x_mean = center.copy()
            sigma = sigma0
            C = np.eye(dim)

            # Evolution path
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Generate initial population
            def sample_population():
                rng = np.random.default_rng()
                z = rng.standard_normal((lambda_, dim))
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                pop = x_mean[None, :] + sigma * (z @ L.T)
                pop = np.clip(pop, -5.0, 5.0)
                return pop

            # Evaluate initial population
            pop = sample_population()
            fvals = np.empty(lambda_)
            for i in range(lambda_):
                if not (budget_ok() and time_ok()):
                    break
                fvals[i] = func(pop[i])
                calls_made += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            if not (budget_ok() and time_ok()):
                break

            # Main CMA-ES loop
            max_gens = 200
            for gen in range(max_gens):
                if not (budget_ok() and time_ok()):
                    break
                if calls_made >= budget - 10:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                pop_sorted = pop[order]
                fvals_sorted = fvals[order]

                # Update mean
                x_mean_new = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

                # Check stagnation
                if np.linalg.norm(x_mean_new - x_mean) < 1e-16:
                    break

                # Update evolution paths
                diff = (x_mean_new - x_mean) / sigma
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-14)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(dim)

                c_inv_sqrt_diff = C_inv_sqrt @ diff
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * c_inv_sqrt_diff

                denom = np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) * chi_n
                h_sigma = 1.0 if (np.linalg.norm(p_sigma) / denom) < 1.4 + 2.0 / (dim + 1) else 0.0

                p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * c_inv_sqrt_diff

                # Update C
                delta_h = (h_sigma - 1) * c_c * (2 - c_c) * np.outer(p_c, p_c)
                C_new = (1 - c_1 - c_mu + delta_h) * C
                C_new += c_1 * np.outer(p_c, p_c)
                y_selected = (pop_sorted[:mu] - x_mean) / sigma
                for i in range(mu):
                    C_new += c_mu * weights[i] * np.outer(y_selected[i], y_selected[i])

                # Ensure C is symmetric positive definite
                C_new = (C_new + C_new.T) / 2.0
                try:
                    eigvals, eigvecs = np.linalg.eigh(C_new)
                    if np.min(eigvals) < 1e-12:
                        eigvals = np.maximum(eigvals, 1e-12)
                        C_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except:
                    C_new = C

                C = C_new
                x_mean = x_mean_new

                # Update sigma
                ratio = np.linalg.norm(p_sigma) / chi_n
                sigma *= np.exp(c_sigma / d_sigma * (ratio - 1.0))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Generate next population
                pop = sample_population()
                fvals = np.empty(lambda_)
                for i in range(lambda_):
                    if not (budget_ok() and time_ok()):
                        break
                    fvals[i] = func(pop[i])
                    calls_made += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()

                if not (budget_ok() and time_ok()):
                    break

                # Check for convergence
                if np.max(fvals_sorted) - np.min(fvals_sorted) < 1e-12:
                    break
                if sigma < 1e-8:
                    break

        # Final polish with coordinate descent
        if best_x is not None and budget_ok() and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma_cd = 0.1
            sigma_min = 1e-6

            max_cd_iters = min(20, max(0, budget - calls_made - 2))
            for cd_iter in range(max_cd_iters):
                if not (budget_ok() and time_ok()):
                    break

                improved = False
                for i in range(dim):
                    if not (budget_ok() and time_ok()):
                        break

                    for direction in (1.0, -1.0):
                        y = x.copy()
                        y[i] += direction * sigma_cd
                        y = np.clip(y, -5.0, 5.0)
                        fy = func(y)
                        calls_made += 1

                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

                if not improved:
                    sigma_cd *= 0.5
                    if sigma_cd < sigma_min:
                        break
                else:
                    sigma_cd = min(sigma_cd * 1.2, 1.0)

        # Ensure valid result
        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made < budget:
                best_f = func(best_x)
                calls_made += 1
            else:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
