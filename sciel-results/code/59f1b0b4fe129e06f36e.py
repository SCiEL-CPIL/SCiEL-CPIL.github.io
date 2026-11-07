"""Rewritten optimizer using CMA-ES for the Rosenbrock function."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with adaptive restarts and boundary handling for Rosenbrock on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # CMA-ES parameters tuned for dim=5
        lambda_ = 4 + int(3 * np.log(dim))  # ~10 for dim=5
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x_clipped = clip(x)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f, x_clipped

        # Initial evaluation at center
        mean = np.zeros(dim)
        f0, x0 = evaluate(mean)
        best_x = x0.copy()
        best_f = f0

        # CMA-ES state
        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        stagnation_counter = 0
        max_stagnation = 12
        prev_best_f = best_f
        restart_count = 0
        max_restarts = 3

        while calls < budget:
            old_mean = mean.copy()
            old_C = C.copy()

            # Generate population
            pop = []
            pop_f = []
            
            remaining = budget - calls
            current_lambda = min(lambda_, remaining)
            
            if current_lambda <= 0:
                break

            # Pre-compute Cholesky factor if C is valid
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))

            for i in range(current_lambda):
                z = rng.standard_normal(dim)
                x_i = mean + sigma * (L @ z)
                f_i, x_i_clipped = evaluate(x_i)
                pop.append(x_i_clipped)
                pop_f.append(f_i)
                
                if calls >= budget:
                    break

            if calls >= budget:
                break

            # Sort by fitness
            sorted_idx = np.argsort(pop_f)
            pop = [pop[i] for i in sorted_idx]
            pop_f = [pop_f[i] for i in sorted_idx]

            # Update mean using weighted recombination
            new_mean = np.zeros(dim)
            for i in range(min(mu, len(pop))):
                new_mean += weights[i] * pop[i]
            
            # Clip mean to stay within bounds
            new_mean = clip(new_mean)
            mean = new_mean

            # Check for stagnation
            if best_f < prev_best_f - 1e-8:
                prev_best_f = best_f
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Adaptive Restart
            if stagnation_counter >= max_stagnation:
                if restart_count < max_restarts:
                    # Restart CMA-ES state centered at best_x with reduced sigma
                    mean = best_x.copy()
                    sigma = max(0.1, sigma * 0.5)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                    C = np.eye(dim)
                    stagnation_counter = 0
                    prev_best_f = best_f
                    restart_count += 1
                else:
                    # If we've restarted too many times, do local search
                    # Try small perturbations around best_x
                    local_sigma = 0.1
                    for _ in range(10):
                        if calls >= budget:
                            break
                        x_try = best_x + local_sigma * rng.standard_normal(dim)
                        x_try = clip(x_try)
                        f_try = func(x_try)
                        calls += 1
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                            local_sigma *= 0.5
                    break
                continue

            # Update evolution paths
            pc = (1 - c_c) * pc + c_c * ((mean - old_mean) / sigma)
            
            try:
                inv_L = np.linalg.inv(L)
                diff_whitened = inv_L @ ((mean - old_mean) / sigma)
            except np.linalg.LinAlgError:
                diff_whitened = (mean - old_mean) / sigma

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff_whitened

            # Update step size (sigma)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)

            # Update covariance matrix
            C_new = (1 - c_1 - c_mu) * old_C
            C_new += c_1 * (np.outer(pc, pc) + (1 - c_c) * np.eye(dim))
            
            for i in range(min(mu, len(pop))):
                diff_i = (pop[i] - old_mean) / sigma
                C_new += c_mu * weights[i] * np.outer(diff_i, diff_i)

            # Ensure C is symmetric positive definite
            C_new = (C_new + C_new.T) / 2
            C_new += 1e-12 * np.eye(dim)
            C = C_new

        return best_x, best_f
# EVOLVE-BLOCK-END
