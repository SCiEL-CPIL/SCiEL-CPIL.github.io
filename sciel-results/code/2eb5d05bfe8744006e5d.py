"""Optimizer for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.perf_counter()
        deadline = t0 + 250.0          # safety margin below the 330s kill
        rng = np.random.default_rng(0)

        # CMA-ES hyperparameters tuned for dim=5
        popsize = 4 + int(round(3 * np.log(dim)))   # 12
        lambda_ = popsize
        mu = lambda_ // 2                            # 6
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w * w)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma

        # Broad start: minimizer is hidden in [-4,4]^5, so sample the whole box.
        # A wide initial sigma lets the first evaluations already land near it.
        x = np.zeros(dim)
        sigma = 3.5

        # Build the initial population and evaluate it (lambda_ calls).
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        P = [np.clip(x + sigma * (B @ (D * rng.standard_normal(dim))), -5, 5)
             for _ in range(lambda_)]
        fvals = np.array([func(p) for p in P])

        # Sort by fitness (ascending).
        order = np.argsort(fvals)
        P = [P[i] for i in order]
        fvals = fvals[order]

        best = P[0].copy()
        bestf = fvals[0]

        def time_ok():
            return time.perf_counter() < deadline

        calls = lambda_
        while calls < budget:
            # Update the mean from the best mu solutions.
            xnew = sum(w[i] * P[i] for i in range(mu))

            # Update the covariance matrix.
            y = (xnew - x) / sigma
            y = B.T @ (D * y)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1)) / chiN) < (1.4 + 2 / (dim + 1))
            pc = ((1 - c_c) * pc + (hsig / damps) * c_c * y) if hsig else (1 - c_c) * pc
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = ((1 + c_1 * delta_hsig - c_1 - c_mu) * C
                 + c_1 * np.outer(pc, pc)
                 + c_mu / sigma ** 2 * sum(w[i] * np.outer((P[i] - x) / sigma,
                                                           (P[i] - x) / sigma) for i in range(mu)))
            # Recompute eigen-decomposition (cheap for dim=5).
            D2, B2 = np.linalg.eigh(C)
            D2 = np.sqrt(np.clip(D2, 1e-12, None))
            B = B2
            D = D2
            x = xnew
            sigma *= np.exp(c_sigma / damps * (np.linalg.norm(ps) / chiN - 1))

            if not time_ok():
                break

            # Generate and evaluate the next population (lambda_ calls).
            P = [np.clip(x + sigma * (B @ (D * rng.standard_normal(dim))), -5, 5)
                 for _ in range(lambda_)]
            calls += lambda_
            fvals = np.array([func(p) for p in P])
            order = np.argsort(fvals)
            P = [P[i] for i in order]
            fvals = fvals[order]
            if fvals[0] < bestf:
                bestf = fvals[0]
                best = P[0].copy()

            if not time_ok():
                break

        return best, bestf
