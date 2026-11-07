"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio optimizer: diversified (1+1)-ES with coordinate refinement and CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320.0  # 320s safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.full(dim, 5.0)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f
                return True
            return False

        def time_left():
            return deadline - time.time() > 1.0

        # --- Phase 1: Quick random sampling to find a good starting point ---
        rng = np.random.default_rng(42)
        calls = 0

        # Sample a few random points to find a decent starting point
        n_samples = min(10, budget)
        for _ in range(n_samples):
            if not time_left() or calls >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            calls += 1
            update_best(x, fx)

        if calls >= budget or not time_left():
            return best_x, best_f

        # --- Phase 2: Diversified (1+1)-ES from the best point ---
        x = best_x.copy()
        fx = best_f
        sigma = 1.5

        # Run (1+1)-ES for a chunk of the budget
        steps1 = min(150, budget - calls)
        for _ in range(steps1):
            if not time_left() or calls >= budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            update_best(y, fy)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.88

        if calls >= budget or not time_left():
            return best_x, best_f

        # --- Phase 3: Coordinate-wise line search refinement ---
        x4 = best_x.copy()
        fx4 = best_f
        for coord in range(dim):
            if not time_left() or calls >= budget:
                break
            for direction in [1, -1]:
                if not time_left() or calls >= budget:
                    break
                step = 0.5
                improved = True
                while improved and time_left() and calls < budget:
                    improved = False
                    for trial in [step, step * 0.5, step * 2.0]:
                        if calls >= budget or not time_left():
                            break
                        trial_x = x4.copy()
                        trial_x[coord] = np.clip(x4[coord] + direction * trial, -5, 5)
                        trial_f = func(trial_x)
                        calls += 1
                        if trial_f < fx4:
                            x4, fx4 = trial_x, trial_f
                            update_best(x4, fx4)
                            improved = True
                            break
                    if improved:
                        step *= 1.5
                    else:
                        break

        if calls >= budget or not time_left():
            return best_x, best_f

        # --- Phase 4: CMA-ES for global refinement ---
        x_mean = best_x.copy()
        sigma_cma = 0.5

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_1 = (2 / ((dim + 1.3)**2 + mu_eff))
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        # Run CMA-ES for remaining budget
        while calls < budget and time_left():
            # Generate population
            offspring = []
            for i in range(pop_size):
                if not time_left() or calls >= budget:
                    break
                z = rng.standard_normal(dim)
                # Sample from multivariate normal using Cholesky of C
                y = x_mean + sigma_cma * (C @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((fy, y))

            if len(offspring) < mu:
                break

            # Sort offspring
            offspring.sort(key=lambda x: x[0])
            selected = offspring[:mu]

            # Update mean
            x_new = np.average([s[1] for s in selected], axis=0, weights=weights)

            # Update covariance matrix
            diff = np.array([s[1] for s in selected]) - x_mean
            delta = np.average(diff, axis=0, weights=weights)

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ delta) / sigma_cma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * delta

            # Update C
            rank_one = np.outer(p_sigma, p_sigma)
            rank_mu = np.array([[np.outer(diff[i], diff[i]) * weights[i] for i in range(mu)]])
            rank_mu = rank_mu.sum(axis=0)

            C = (1 + c_1 * (mu_eff - 1) - c_mu) * C + c_mu * rank_mu + c_1 * (1 - mu_eff) * rank_one

            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            sigma_cma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))

            x_mean = x_new

            # Update best
            for fy, y in selected:
                update_best(y, fy)

        return best_x, best_f

# EVOLVE-BLOCK-END
