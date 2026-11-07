"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility within the instance
        rng = np.random.default_rng(42)

        # Multi-start: try a few initial points, pick the best
        n_starts = min(5, budget // 50)
        starts = []
        for _ in range(n_starts):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            starts.append((x, fx))

        # Pick the best start
        starts.sort(key=lambda t: t[1])
        x_best, fx_best = starts[0]

        # CMA-ES parameters
        # Standard CMA-ES for the Katsuura function in 5D
        # The function is multimodal with many local minima, so we use
        # a relatively large initial sigma and multiple restarts if stuck

        lambda_ = 4 * int(np.ceil(3 + np.log(dim)))  # population size
        mu = lambda_ // 2  # number of selected solutions

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = int(np.sum(weights[:min(mu, int(1.0 / np.sum(weights ** 2)))]) ** 2)

        # Initial step size
        sigma = 1.5

        # CMA-ES state
        x_mean = x_best.copy()
        f_best = fx_best
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + dim / mueff) / (dim + 4 + 2 / mueff)
        c1 = (mueff + 2) / ((dim + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        calls_made = n_starts
        no_improve = 0
        max_no_improve = 80  # restart if no improvement for this many steps

        for gen in range(budget - n_starts):
            # Time check
            if time.time() > deadline:
                break

            # Generate lambda_ offspring
            D = np.sqrt(np.diag(C))
            B = np.linalg.inv(np.diag(D)) @ C
            # Actually, let's use the standard CMA-ES update
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            y = []
            fy = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                xi = x_mean + sigma * (L @ z)
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_made += 1
                y.append(xi)
                fy.append(fi)

            if calls_made >= budget:
                break

            # Sort by fitness
            idx = np.argsort(fy)
            y_sorted = [y[i] for i in idx]
            fy_sorted = [fy[i] for i in idx]

            # Update mean
            x_old_mean = x_mean.copy()
            x_mean = sum(weights[i] * y_sorted[i] for i in range(mu))

            # Check for improvement
            if fy_sorted[0] < f_best:
                f_best = fy_sorted[0]
                x_best = y_sorted[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Update evolution paths
            y_invC = np.linalg.solve(C, x_mean - x_old_mean)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (y_invC / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1))) / np.sqrt(dim) < (1.4 + 2 / (dim + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (y_invC / sigma)

            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + delta_hsig - 1) * np.eye(dim)
            C += cmu * sum(weights[i] * np.outer(y_invC_i, y_invC_i)
                           for i, y_invC_i in enumerate([np.linalg.solve(C, y_sorted[i] - x_old_mean) for i in range(mu)]))

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Restart if stuck
            if no_improve >= max_no_improve:
                # Restart from best point with larger sigma
                x_mean = x_best.copy()
                sigma = max(sigma * 2.0, 2.0)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                no_improve = 0
                max_no_improve = int(max_no_improve * 1.5)

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return x_best, f_best
# EVOLVE-BLOCK-END
