"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_ // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 - cc + 2 * min(1, (mueff - 2) / (dim + 3)) * max(0, 1 - 1 / (dim + 1))) / (
            (dim + 1.3) ** 2 + mueff
        )
        cmu = min(
            1 - c1,
            2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff),
        )
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_chol = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial evaluation
        evaluate(x_mean)

        for generation in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                pop[i] = x_mean + sigma * (C_chol @ z)
                pop[i] = np.clip(pop[i], -5, 5)

            # Evaluate population
            fitness = np.array([evaluate(pop[i]) for i in range(lambda_)])

            # Sort
            idx = np.argsort(fitness)
            x_sorted = pop[idx]

            # Compute y vectors (normalized steps)
            y_vecs = (x_sorted[:mu] - x_mean) / sigma

            # Update step size control variable
            C_inv_sqrt = np.linalg.inv(C_chol)
            y_mean = np.dot(w, y_vecs)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ y_mean)

            # Update evolution path
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * y_mean

            # Hessian update
            delta_sig = (np.linalg.norm(ps) / chiN - 1) * (1 + cs * (1 - cs))
            ps_norm = np.linalg.norm(ps)
            threshold = (1.4 + 2 / (dim + 1)) * chiN
            hsig = 1 if ps_norm / np.sqrt(1 - (1 - cs) ** (2 * (generation + 1) / (dim + mueff))) < threshold else 0

            # Update covariance matrix
            delta_h = (1 - hsig) * (1 - cc)
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                rankmu += w[i] * np.outer(y_vecs[i], y_vecs[i])

            C = (1 + c1 * delta_h + c1 * delta_sig - c1 - cmu) * C + c1 * (rank1 + delta_h * np.eye(dim) - C) + cmu * rankmu

            # Ensure positive definite
            C = (C + C.T) / 2
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_chol = np.eye(dim)

            # Update mean
            x_mean = x_mean + np.dot(w, x_sorted[:mu] - x_mean)
            x_mean = np.clip(x_mean, -5, 5)

            # Update sigma
            sigma = sigma * np.exp(cs / damps * (ps_norm / chiN - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Check termination
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
