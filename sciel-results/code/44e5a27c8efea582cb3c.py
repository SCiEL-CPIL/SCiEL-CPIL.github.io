"""Rewritten optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time-guarded evaluation budget."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        # Start near center for f6 (attractive sector)
        x = np.zeros(n)
        f = func(x)
        best_x = x.copy()
        best_f = f
        calls = 1

        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        # CMA-ES parameters
        mu = max(1, n // 2)  # number of parents
        lambda_ = 2 * mu     # population size
        wi = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        wi /= np.sum(wi)
        mueff = np.sum(wi) ** 2 / np.sum(wi ** 2)

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        # Precompute Cholesky decomposition of C
        try:
            Ch = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            Ch = np.eye(n)

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            xs = []
            fs = []
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                zi = rng.standard_normal(n)
                yi = x + sigma * (Ch @ zi)
                xi = np.clip(yi, -5, 5)
                fi = func(xi)
                calls += 1
                ys.append(yi)
                xs.append(xi)
                fs.append(fi)

            if not ys:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order[:mu]]
            xs = [xs[i] for i in order[:mu]]
            fs = [fs[i] for i in order[:mu]]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()

            # Mean
            m = np.sum([wi[i] * ys[i] for i in range(mu)], axis=0)

            # Update covariance matrix adaptation
            d = 1 + 2 * max(0, (cs + 1) / (1 + cs) - 1)  # damps cs
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Ch @ ((m - x) / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls - budget // lambda_ + 1)))) < (1 + cc) * np.sqrt(n + 1) / np.sqrt(n - 1)
            hsig = 1 if hsig else 0

            pc = (1 - cc) * pc + hsig * cc * (m - x)
            delta_hs = (1 - hsig) * hsig * (1 + cc)

            C = (1 + c1 * delta_hs - c1 - cmu) * C + c1 * (np.outer(pc, pc) - C)
            for i in range(mu):
                C += cmu * wi[i] * np.outer(ys[i] - x, ys[i] - x)

            # Update sigma
            sigma *= np.exp(cs / (5 * n) * (np.linalg.norm(ps) / np.sqrt(n) - 1))

            # Update mean
            x = m

            # Ensure Cholesky is valid
            try:
                Ch = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                Ch = np.eye(n)

            # Adaptive sigma scaling based on progress
            if sigma < 0.001:
                sigma = 0.001
            elif sigma > 10:
                sigma = 10

        return best_x, best_f
# EVOLVE-BLOCK-END
