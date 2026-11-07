"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """CMA-ES with Powell polish for BBOB-style Rosenbrock in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls = 0

        def safe_eval(x):
            nonlocal calls, best_x, best_fx
            if not np.all(np.isfinite(x)) or x.shape != (n,):
                return np.inf
            if calls >= budget:
                return np.inf
            val = func(x)
            calls += 1
            if np.isfinite(val) and val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        # Phase 1: Initial Exploration - find a good basin
        n_initial = min(20, budget // 5)
        for i in range(n_initial):
            if time.time() > deadline or calls >= budget:
                break
            if i == 0:
                x = np.zeros(n)
            elif i == 1:
                x = np.ones(n) * 0.5
            elif i == 2:
                x = np.ones(n)
            elif i == 3:
                x = np.ones(n) * 1.0
            else:
                x = rng.uniform(-4, 4, n)
            safe_eval(x)

        if best_x is None:
            best_x = np.zeros(n)
            safe_eval(best_x)

        # Phase 2: CMA-ES optimization
        pop_size = 4 + int(3 * np.log(n))  # ~9 for n=5
        pop_size = max(pop_size, 6)

        x_mean = best_x.copy()
        sigma = 2.0  # larger initial step for Rosenbrock

        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 - mu_eff + n) / (n + 4 - mu_eff)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        remaining_budget = budget - calls
        max_generations = remaining_budget // pop_size
        max_generations = max(max_generations, 10)

        for gen in range(max_generations):
            if time.time() > deadline:
                break
            if calls + pop_size > budget:
                num_evals = budget - calls
                if num_evals <= 0:
                    break
            else:
                num_evals = pop_size

            # Generate population
            offspring = []
            for i in range(num_evals):
                z = rng.standard_normal(n)
                x_i = x_mean + sigma * (B @ (D * z))
                offspring.append(x_i)

            # Evaluate offspring
            f_values = []
            for x_i in offspring:
                if time.time() > deadline or calls >= budget:
                    f_values.append(np.inf)
                    continue
                f_val = safe_eval(x_i)
                f_values.append(f_val)

            if len(f_values) < 2:
                break

            # Sort by fitness
            sorted_indices = np.argsort(f_values)
            selected_indices = sorted_indices[:min(mu, len(offspring))]

            if len(selected_indices) == 0:
                break

            # Update mean
            x_new_mean = np.zeros(n)
            for i, idx in enumerate(selected_indices):
                x_new_mean += weights[i] * offspring[idx]

            # Update covariance matrix
            y_vec = (x_new_mean - x_mean) / sigma if sigma > 1e-20 else np.zeros(n)

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) / chi_n) < 1.4 + 2 / (n + 1)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec

            delta_h = (1 - h_sigma) * c_c * (1 - c_c)
            rank_mu = np.zeros((n, n))
            for i, idx in enumerate(selected_indices):
                y_i = (offspring[idx] - x_mean) / sigma if sigma > 1e-20 else np.zeros(n)
                rank_mu += weights[i] * np.outer(y_i, y_i)

            if h_sigma:
                rank_one = np.outer(p_c, p_c)
            else:
                rank_one = np.outer(p_c, p_c) + delta_h * np.outer(p_c, p_c)

            # Update C
            C = (1 + c_1 * (rank_one - np.eye(n)) + c_mu * rank_mu) * C

            # Cholesky decomposition for B and D
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                B = L.T
                D = np.diag(L)
            except np.linalg.LinAlgError:
                # Fallback: eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs
                D = np.sqrt(eigvals)

            x_mean = x_new_mean

            # Adapt sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Stagnation check: if sigma gets too small, increase it
            if sigma < 1e-6:
                sigma = 0.1

            # Early termination if convergence is good
            if np.all(np.abs(y_vec) < 1e-8) and gen > 5:
                break

        # Phase 3: Powell's method polish from best point
        if best_x is not None and time.time() < deadline and calls < budget:
            def nm_wrapper(x):
                nonlocal calls, best_x, best_fx
                if calls >= budget:
                    return np.inf
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                calls += 1
                if np.isfinite(val) and val < best_fx:
                    best_fx = val
                    best_x = x_clipped.copy()
                return val

            # Try Powell from best point
            try:
                res = scipy.optimize.minimize(
                    nm_wrapper,
                    best_x.copy(),
                    method='Powell',
                    options={
                        'maxiter': min(budget - calls, 100),
                        'xtol': 1e-12,
                        'ftol': 1e-14,
                        'disp': False
                    }
                )
            except Exception:
                pass

            # Try Nelder-Mead from best point if budget remains
            if time.time() < deadline and calls < budget and budget - calls >= 10:
                try:
                    res = scipy.optimize.minimize(
                        nm_wrapper,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(budget - calls, 100),
                            'xatol': 1e-12,
                            'fatol': 1e-14,
                            'disp': False
                        }
                    )
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(n)
            safe_eval(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
