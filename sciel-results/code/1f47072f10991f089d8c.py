"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """IPOP-CMA-ES for BBOB f18 with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill

        rng = np.random.default_rng(42)

        # --- Phase 1: brief random search for early AOCC coverage ---
        n_random = 10
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        for i in range(n_random):
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 2: IPOP-CMA-ES ---
        # Initial population size
        lam = max(4, int(4 + 3 * np.log(n)))  # ~7 for n=5
        # Cap lambda so we have enough budget for multiple generations
        if lam > budget // 20:
            lam = max(4, budget // 20)

        # CMA-ES parameters
        c1 = (4 + np.log(n) / n) / (n**2 + 2)
        cmu = min(1 - c1, 2 * (n + 1.7 - n) / (n**2 + 4))
        c_sigma = (n + 1.8) / (n + 9)
        d_sigma = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        c_cs = (n + 1.5) / (n + 9 + 2)
        d_cs = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + c_cs

        # Initialize CMA-ES state
        x_mean = best_x.copy()
        sigma = 2.0  # initial step size
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n_val = chi_n

        # Track best
        best_x_es = best_x.copy()
        best_f_es = best_f

        # IPOP: population size will shrink when stagnation detected
        no_improve = 0
        min_lam = 4
        lam_current = lam

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Check if we should shrink population (IPOP)
            if no_improve > 10 * lam_current and lam_current > min_lam:
                lam_current = max(min_lam, lam_current // 2)
                no_improve = 0
                # Resample sigma
                sigma *= 0.5

            # Generate population
            D = np.linalg.cholesky(C)
            population = []
            fitnesses = []

            for i in range(lam_current):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = x_mean + sigma * D @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                population.append(y)
                fitnesses.append(fy)

            if len(population) == 0:
                break

            population = np.array(population)
            fitnesses = np.array(fitnesses)

            # Sort by fitness
            order = np.argsort(fitnesses)
            population = population[order]
            fitnesses = fitnesses[order]

            mu = lam_current // 2

            # Update best
            if fitnesses[0] < best_f_es:
                best_f_es = fitnesses[0]
                best_x_es = population[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Update mean (weighted recombination)
            weights = np.zeros(mu)
            weights[0] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)).sum()
            for i in range(1, mu):
                weights[i] = weights[0] - np.log(i + 1)
            weights = weights / weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)

            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * population[:mu], axis=0)

            # Update step-size control (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (D @ (x_mean - x_mean_old))
            if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (n + 1)) * np.sqrt((1 - c_sigma) / (1 + c_sigma)):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance matrix
            v = (x_mean - x_mean_old) / sigma
            pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mu_eff) * v
            outer_v = np.outer(v, v)
            delta_heaviside = 1 if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (n + 1)) * np.sqrt((1 - c_sigma) / (1 + c_sigma)) else 0
            C = (1 - c1 * (1 + cmu - c1)) * C + c1 * outer_v + cmu * np.sum(weights[i] * np.outer(population[i] - x_mean_old, population[i] - x_mean_old) / sigma**2 for i in range(mu))
            C += cmu * delta_heaviside * (np.outer(pc, pc) - C) * c1 * (1 - c1) / c1
            # Ensure positive definiteness
            C = 0.5 * (C + C.T)

            # Check for convergence
            if sigma < 1e-12 or np.all(np.abs(x_mean - best_x_es) < 1e-10):
                break

            # If we've used most of the budget, break
            if calls >= budget:
                break

        return best_x_es, best_f_es
# EVOLVE-BLOCK-END
