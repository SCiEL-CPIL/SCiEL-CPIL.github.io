"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive CMA-ES variant with restarts for Rastrigin on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf

        # We'll do multiple restarts of CMA-ES
        # Each restart gets a portion of the budget
        # For Rastrigin, which has many local minima, restarts help escape

        # CMA-ES parameters
        lambda_pop = 4 + int(3 * np.log(n))  # population size
        mu = lambda_pop // 2  # number of recombination points
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()

        # Step size adaptation (CSA)
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs

        # Update path parameters
        hsig = (1.4 + 2 / (n + 1))  # threshold

        rng = np.random.default_rng(42)

        # Number of restarts
        # For Rastrigin in 5D, multiple restarts are beneficial
        # Let's do 5 restarts with decreasing budget
        n_restarts = 5
        budgets = []
        remaining = total_budget
        for i in range(n_restarts):
            frac = 1.0 / (i + 1) - (1.0 / (i + 2)) if i < n_restarts - 1 else 1.0 / n_restarts
            b = int(remaining * (0.5 if i == 0 else 0.3))
            if i == n_restarts - 1:
                b = remaining
            budgets.append(b)
            remaining -= b

        # Ensure at least some budget for each
        for i in range(len(budgets)):
            if budgets[i] < 10:
                budgets[i] = 10

        # Adjust to fit total
        while sum(budgets) > total_budget:
            max_idx = np.argmax(budgets)
            budgets[max_idx] -= 1
        while sum(budgets) < total_budget:
            min_idx = np.argmin(budgets)
            budgets[min_idx] += 1

        for restart_idx, budget in enumerate(budgets):
            if time.time() > deadline:
                break

            # Initial point
            if best_x is not None and restart_idx > 0:
                # Perturb around best known
                x = np.clip(best_x + rng.standard_normal(n) * 2.0, -5, 5)
            else:
                x = rng.uniform(-5, 5, n)

            fx = func(x)
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            if budget < 2:
                continue

            # CMA-ES parameters
            sigma = 1.5  # initial step size
            C = np.eye(n)  # covariance matrix
            pc = np.zeros(n)  # evolution path for covariance
            ps = np.zeros(n)  # evolution path for step size

            # Precompute sqrt of C
            # Use Cholesky for efficiency
            try:
                sqrtC = np.linalg.cholesky(C)
            except:
                sqrtC = np.eye(n)

            # Normalization constant for CSA
            chisigma = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            calls_used = 1  # initial evaluation

            for gen in range(budget - 1):
                if time.time() > deadline:
                    return best_x, best_f
                if calls_used >= total_budget:
                    break

                # Generate population
                offspring = []
                offspring_f = []

                for k in range(lambda_pop):
                    # Sample
                    z = rng.standard_normal(n)
                    y = x + sigma * (sqrtC @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    offspring.append(y)
                    offspring_f.append(fy)

                    if fy < best_f:
                        best_f, best_x = fy, y.copy()

                    if calls_used >= total_budget:
                        break

                if len(offspring) < mu:
                    break

                # Sort by fitness
                idx = np.argsort(offspring_f)
                selected = [offspring[i] for i in idx[:mu]]
                selected_f = [offspring_f[i] for i in idx[:mu]]

                # Weighted mean
                m = np.average(selected, axis=0, weights=weights)

                # Update evolution paths
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff/C) * (m - x)/sigma
                diff = (m - x) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * diff

                # Check for sigma adaptation
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (hsig + 0.3) * chisigma:
                    sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chisigma - 1))

                # Update pc
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * diff

                # Update covariance matrix
                # C = (1 + mu_sigma*cs*(2-cs) - cs) * C + mu_sigma * outer(pc, pc)
                # where mu_sigma = 1 - exp(-cs/mueff * ...)
                # Simplified: use rank-1 + rank-mu update
                mu_sigma = 1 - np.exp(-cs / mueff * (1 + weights.sum()))  # approximation
                # Actually, standard CMA-ES uses:
                # C = (1 + 2*mu_sigma*cs*(2-cs) - cs) * C + mu_sigma * outer(pc, pc)
                # + sum_k weights_k * outer(y_k - x, y_k - x) / sigma^2

                # Let's use a simpler but effective update
                # Rank-1 update from pc
                C = (1 - cs) * C + (1 + 2 * mu_sigma - cs) * np.outer(pc, pc)

                # Rank-mu update
                for k in range(mu):
                    d = (selected[k] - x) / sigma
                    C += weights[k] * np.outer(d, d)

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update mean
                x = m

                # Update Cholesky
                try:
                    sqrtC = np.linalg.cholesky(C)
                except:
                    # Add small regularization
                    C += 1e-8 * np.eye(n)
                    try:
                        sqrtC = np.linalg.cholesky(C)
                    except:
                        sqrtC = np.eye(n)

        return best_x, best_f
# EVOLVE-BLOCK-END
