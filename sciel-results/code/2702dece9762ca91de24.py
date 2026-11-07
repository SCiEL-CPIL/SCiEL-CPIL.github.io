"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: CMA-ES
        # Initialize
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        lambda_ = dim + int(np.floor(3 * np.log(dim)))  # population size
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        cs = (1 + 2 * max(0, np.sqrt((lambda_ / dim - 1)) - 1)) / (dim + 1 + 2 * lambda_ / dim)
        cmu = min(2 - cc, 2 / ((dim + 1.3) ** 2 - dim))
        c1 = (2 + 2 * cmu) / ((dim + 1.7) ** 2 + lambda_)
        cmu_eff = min(cmu, c1)

        calls_made = 1
        no_improve = 0

        while calls_made < self.budget:
            # Check time
            if time.time() > deadline - 50:
                break

            # Generate population
            population = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x + sigma * (C @ z)
                y = np.clip(y, -5, 5)
                population.append(y)

            # Evaluate population
            for y in population:
                if calls_made >= self.budget:
                    break
                fy = func(y)
                calls_made += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    no_improve = 0
                else:
                    no_improve += 1

            if calls_made >= self.budget:
                break

            # Sort by fitness
            fvals = [func(y) for y in []]  # already evaluated above, need to store
            # Actually let's restructure: store (y, fy) pairs
            # Let me redo this properly

            break  # placeholder, will fix below

        # Let me rewrite the CMA-ES loop properly
        x = best_x
        fx = best_fx
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        cs = (1 + 2 * max(0, np.sqrt((lambda_ / dim - 1)) - 1)) / (dim + 1 + 2 * lambda_ / dim)
        cmu = min(2 - cc, 2 / ((dim + 1.3) ** 2 - dim))
        c1 = (2 + 2 * cmu) / ((dim + 1.7) ** 2 + lambda_)
        cmu_eff = min(cmu, c1)

        calls_made = 1
        best_x = x.copy()
        best_fx = fx
        no_improve_count = 0

        while calls_made < self.budget and time.time() < deadline - 60:
            # Generate population
            offspring = []
            for i in range(lambda_):
                if calls_made >= self.budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (C @ z)
                y = np.clip(y, -5, 5)
                offspring.append((y, None))
                calls_made += 1
                fy = func(y)
                offspring[-1] = (y, fy)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            if len(offspring) < 2:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[1])
            mu = lambda_ // 2  # number of recombination weights
            # Weights (linear)
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            # Update mean
            x_new = np.sum([weights[i] * offspring[i][0] for i in range(mu)], axis=0)

            # Update step size control (CSA)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C @ (x_new - x))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls_made + 1))) < \
                   (1.4 + 2 / (dim + 1)) * cc

            # Update covariance matrix
            pc = (1 - cc) * pc + hsig * cc * (x_new - x)
            delta = (2 * hsig - 1) * c1
            C = (1 - c1 - cmu_eff + delta) * C
            C += c1 * np.outer(pc, pc)
            for i in range(mu):
                y_i = (offspring[i][0] - x) / sigma
                C += cmu_eff * weights[i] * np.outer(y_i, y_i)

            x = x_new
            sigma *= np.exp(cs / (dim + 1.7) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** calls_made) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Check for convergence
            if np.linalg.norm(x_new - x) < 1e-10:
                break

        # Phase 2: Nelder-Mead polish
        if calls_made < self.budget and time.time() < deadline - 20:
            # Use scipy's minimize for Nelder-Mead
            from scipy.optimize import minimize

            # Set up initial simplex
            x0 = best_x.copy()
            f0 = best_fx
            # Create simplex
            simplex = [x0]
            for i in range(dim):
                xi = x0.copy()
                step = 0.1 if abs(x0[i]) < 1 else 0.01
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            # Manual Nelder-Mead to avoid extra imports issues
            # Actually scipy is available, let's use it
            try:
                result = minimize(
                    func, x0, method='Nelder-Mead',
                    options={'maxiter': self.budget - calls_made - 1,
                             'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
                calls_made = self.budget  # assume we used remaining budget
            except Exception:
                pass

        return best_x, best_fx

# EVOLVE-BLOCK-END
