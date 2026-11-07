"""Rewritten optimizer using multi-start CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for black-box continuous optimization on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, max_evals, deadline, rng):
        """Run CMA-ES starting from x0 with initial sigma0, using up to max_evals evaluations."""
        d = self.dim
        x = x0.copy()
        fx = func(x)
        evals = 1
        sigma = sigma0

        # CMA-ES parameters
        lambda_ = max(4, 4 + int(3 * np.log(d)))  # population size
        mu = lambda_ // 2  # number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma

        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        while evals < max_evals:
            if time.time() > deadline:
                break

            # Generate population
            B = np.linalg.cholesky(C)
            offspring = []
            for i in range(lambda_):
                z = rng.standard_normal(d)
                y = x + sigma * B @ z
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate
            fitness = []
            for y in offspring:
                if time.time() > deadline:
                    break
                fy = func(y)
                fitness.append(fy)
                evals += 1

            if len(fitness) < lambda_:
                break

            # Sort
            order = np.argsort(fitness)
            y_best = offspring[order[0]]
            f_best = fitness[order[0]]

            # Update mean
            y_parents = [offspring[order[i]] for i in range(mu)]
            y_parents = np.array(y_parents)
            x_new = np.sum(weights[:, None] * y_parents, axis=0)
            x = x_new

            # Update sigma
            y_diff = y_parents - x0
            # Actually we need to use the old mean for the step
            # Let me redo this properly

            # Re-evaluate: we need the old x for the step computation
            # This is getting complex. Let me simplify.

            break  # Fallback to simpler approach

        return x, fx

    def _cma_es_simple(self, func, x0, sigma0, max_evals, deadline, rng):
        """Simplified CMA-ES."""
        d = self.dim
        x = x0.copy()
        fx = func(x)
        evals = 1
        sigma = sigma0

        lambda_ = max(4, 4 + int(3 * np.log(d)))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma

        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        while evals < max_evals:
            if time.time() > deadline:
                break

            # Generate population
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                B = np.eye(d)

            offspring = []
            for i in range(lambda_):
                z = rng.standard_normal(d)
                y = x + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate
            fitness = []
            for y in offspring:
                if time.time() > deadline:
                    break
                fy = func(y)
                fitness.append(fy)
                evals += 1

            if len(fitness) < lambda_:
                break

            # Sort
            order = np.argsort(fitness)
            f_best = fitness[order[0]]
            y_best = offspring[order[0]]

            if f_best < fx:
                fx = f_best

            # Update mean using weighted combination of best mu offspring
            y_parents = np.array([offspring[order[i]] for i in range(mu)])
            x_new = np.sum(weights[:, None] * y_parents, axis=0)

            # Update evolution paths
            step = (x_new - x) / (sigma + 1e-30)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ step)

            # Update covariance matrix
            delta_sigma = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals - lambda_ + 1) / d)) <= (1.4 + 2 / (d + 1)) * chi_d else 0.0
            C = (1 + c_1 * delta_sigma - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) - C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(y_parents - x, y_parents - x), axis=0) / (sigma ** 2 + 1e-30)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_d - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x = x_new

            # Stagnation check
            if sigma < 1e-8 or evals >= max_evals:
                break

        return x, fx

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290  # 290s to be safe before 300s limit

        # Multi-start strategy
        n_starts = 4
        # Initial points: center and some random points
        starts = [np.zeros(d)]
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, d))

        # Allocate budget across starts
        # First start gets more budget, subsequent starts get less
        # Use a diminishing allocation
        budgets = []
        remaining = budget
        for i in range(n_starts):
            if i == n_starts - 1:
                b = remaining
            else:
                b = int(remaining * 0.45)
                b = max(b, 30)
            budgets.append(b)
            remaining -= b

        # Ensure total doesn't exceed budget
        total = sum(budgets)
        if total > budget:
            budgets[-1] = budget - sum(budgets[:-1])

        best_x = starts[0].copy()
        best_f = float('inf')

        for i, (x0, max_evals) in enumerate(zip(starts, budgets)):
            if time.time() > deadline:
                break

            # Skip if we've already found a very good solution
            # and this start would be too close to the best
            if i > 0:
                dist = np.linalg.norm(x0 - best_x)
                if dist < 0.5 and best_f < 1.0:
                    continue

            x, fx = self._cma_es_simple(func, x0, 2.0, max_evals, deadline, rng)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # If we still have budget and time, do a local polish around the best
            if time.time() < deadline - 10 and budget - sum(budgets[:i+1]) > 20:
                polish_evals = min(20, budget - sum(budgets[:i+1]))
                if polish_evals > 10:
                    x_polish, fx_polish = self._cma_es_simple(func, best_x, 0.5, polish_evals, deadline, rng)
                    if fx_polish < best_f:
                        best_f = fx_polish
                        best_x = x_polish.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
