"""Optimized black-box continuous optimizer using CMA-ES with restarts."""
import numpy as np
import time


class Optimizer:
    """
    CMA-ES based optimizer with adaptive restarts for BBOB-style functions.
    Uses population-based search with covariance adaptation for efficient
    exploration of multimodal and rotated landscapes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = evaluate(best_x)

        # CMA-ES parameters
        n = self.dim
        pop_size = max(4, int(4 + 3 * np.log(n)))  # lambda = 4 + 3*log(n)
        # For n=5: pop_size = 4 + 3*1.609 = 8 (integer)
        pop_size = 8

        # Run CMA-ES with potential restarts
        remaining = self.budget - calls_made
        if remaining <= 0 or time.time() > deadline:
            return best_x, best_f

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_iter):
            nonlocal best_x, best_f, calls_made
            nonlocal deadline, start_time

            n = self.dim
            lambda_ = pop_size
            mu = lambda_ // 2  # number of weights

            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            # Learning rates
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c1 = (2 + 2 * mueff / (n + 1)) / ((n + 2) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            # State
            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Population storage
            x_pop = np.zeros((lambda_, n))
            f_pop = np.zeros(lambda_)

            for iteration in range(max_iter):
                if time.time() > deadline or calls_made >= self.budget:
                    break

                # Sample population
                for i in range(lambda_):
                    if time.time() > deadline or calls_made >= self.budget:
                        break
                    # Sample from N(x_mean, sigma^2 * C)
                    z = np.random.randn(n)
                    # Cholesky decomposition of C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # If C is not positive definite, reset it
                        C = np.eye(n)
                        L = np.eye(n)
                    x_pop[i] = x_mean + sigma * (L @ z)
                    x_pop[i] = np.clip(x_pop[i], -5.0, 5.0)
                    f_pop[i] = evaluate(x_pop[i])

                # Sort by fitness
                idx = np.argsort(f_pop)
                x_pop = x_pop[idx]
                f_pop = f_pop[idx]

                # Update mean
                x_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * x_pop[:mu], axis=0)

                # Update step-size control
                ps = (1 - cs) * ps + cs * np.sqrt(mueff) / (sigma * np.sqrt(1 - (1 - cs) ** 2)) * (L @ (x_mean - x_old))
                # Simplified: use the standard formula
                cs = (4 + mueff / n) / (n + 4 + 3 * mueff / n)
                ps = (1 - cs) * ps + cs * np.sqrt(mueff) / (sigma) * (x_mean - x_old)

                # Update sigma
                sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 50.0)

                # Update covariance matrix
                hc = (1 - cc) ** (-iteration) - 1  # Approximation
                # More accurate: hc = 1 - (1 - cc) ** (1 / (cc * (1 + cc)))
                hc = 1 - (1 - cc) ** (1 / max(cc, 1e-10))
                pc = (1 - cc) * pc + hc * cs * np.sqrt(mueff) / (sigma) * (x_mean - x_old)

                delta_hc = (1 + hc) * cc * (1 - cc)
                C = (1 + c1 * delta_hc - c1 - cmu) * C + c1 * np.outer(pc, pc)
                C += cmu * sigma ** 2 / (1 - (1 - cs) ** 2) * np.sum(
                    weights[:, None] * np.outer(x_pop[:mu] - x_old, x_pop[:mu] - x_old), axis=0) / sigma ** 2
                # Simplified covariance update
                C = (1 + c1 * delta_hc - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu):
                    yi = (x_pop[i] - x_old) / sigma
                    C += cmu * weights[i] * np.outer(yi, yi)

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Regularize if needed
                eigvals = np.linalg.eigvalsh(C)
                if np.any(eigvals < 1e-10):
                    C = C + 1e-8 * np.eye(n)

                # Stop condition: if sigma is very small or no improvement
                if iteration > 10 and f_pop[0] > best_f + 1e-8:
                    # Check for stagnation
                    if iteration % 20 == 0:
                        if np.all(np.abs(x_pop[:mu] - x_mean) < 1e-4 * sigma):
                            break

            return x_mean, f_pop[0]

        # Run CMA-ES with multiple restarts if needed
        rng = np.random.default_rng(42)
        num_restarts = 3
        budget_per_restart = remaining // num_restarts

        for r in range(num_restarts):
            if time.time() > deadline or calls_made >= self.budget:
                break

            if r == 0:
                x0 = best_x.copy()
                sigma0 = 2.0
            else:
                # Restart from a random point near current best
                x0 = np.clip(best_x + rng.normal(0, 1.0, self.dim), -5, 5)
                sigma0 = 2.0

            max_iter = budget_per_restart // pop_size + 1
            x_mean, f_best = cma_es(x0, sigma0, max_iter)

            if f_best < best_f:
                best_f = f_best
                best_x = x_mean.copy()

        # Final polish: local search around best point
        if calls_made < self.budget and time.time() < deadline:
            # Coordinate descent polish
            step = 0.1
            for _ in range(20):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                improved = False
                for i in range(self.dim):
                    for sign in [1, -1]:
                        x_new = best_x.copy()
                        x_new[i] += sign * step
                        x_new = np.clip(x_new, -5.0, 5.0)
                        if np.allclose(x_new, best_x):
                            continue
                        fx_new = evaluate(x_new)
                        if fx_new < best_f:
                            best_x = x_new
                            best_f = fx_new
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        return best_x, best_f
