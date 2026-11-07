"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_step(self, func, x, sigma, C, rng, deadline, max_iter):
        """Run CMA-ES from x with covariance C. Returns (best_x, best_fx, calls_used)."""
        n = self.dim
        calls = 0
        best_x = x.copy()
        best_fx = func(x)
        calls += 1
        sigma_min = 1e-10
        sigma_max = 50.0

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))
        mu = lambda_ // 2
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 - cc + mueff / (n + 2)) / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        ps = np.zeros(n)
        pc = np.zeros(n)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        damps = min(damps, 2)

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break

            # Generate lambda_ points
            B = rng.standard_normal((lambda_, n))
            # Apply covariance matrix square root (Cholesky)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
                C = np.eye(n)
            Y = (B @ L.T).T  # (lambda_, n)
            Y = x + sigma * Y
            Y = np.clip(Y, -5, 5)

            # Evaluate
            fvals = np.array([func(y) for y in Y])
            calls += lambda_

            # Sort
            idx = np.argsort(fvals)
            Y_sorted = Y[idx]
            fvals_sorted = fvals[idx]

            # Update best
            if fvals_sorted[0] < best_fx:
                best_fx = fvals_sorted[0]
                best_x = Y_sorted[0].copy()

            # Mean update
            m_new = np.sum(w[:, None] * Y_sorted[:mu], axis=0)

            # Step size adaptation (1/5 rule-like)
            successes = np.sum(fvals_sorted[:mu] < best_fx)
            if successes > 0:
                sigma *= np.exp(cs / damps * (successes / mu - 1 / 5))
            sigma = np.clip(sigma, sigma_min, sigma_max)

            # Evolution path updates
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (L @ (m_new - x))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                pc = (1 - cc) * pc + cc / sigma * (L @ (m_new - x))

            # Covariance update
            C = (1 + c1 * cc * (1 - cc) - c1 - cmu) * C + c1 * np.outer(m_new - x, m_new - x)
            C += cmu * np.sum(w[i] * np.outer(Y_sorted[i] - x, Y_sorted[i] - x) for i in range(mu))

            # Ensure symmetry
            C = (C + C.T) / 2

            x = m_new

        return best_x, best_fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        total_calls = 0

        # Phase 1: Random initialization with several starts
        n_init = 8
        for i in range(n_init):
            if time.time() > deadline or total_calls >= self.budget:
                break
            x0 = rng.uniform(-5, 5, self.dim)
            fx0 = func(x0)
            total_calls += 1
            if fx0 < best_fx:
                best_fx = fx0
                best_x = x0.copy()

        # Phase 2: CMA-ES from best point
        remaining = self.budget - total_calls
        if remaining > 10:
            C = np.eye(self.dim)
            bx, bfx, c = self._cma_step(func, best_x, 2.0, C, rng, deadline, remaining)
            total_calls += c
            if bfx < best_fx:
                best_fx = bfx
                best_x = bx

        # Phase 3: Local refinement with (1+1)-ES from best
        remaining = self.budget - total_calls
        if remaining > 5 and best_x is not None:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.5
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                if sigma < 1e-8:
                    break

        # Phase 4: If budget remains, try another CMA-ES restart from a different point
        remaining = self.budget - total_calls
        if remaining > 20:
            x0 = rng.uniform(-5, 5, self.dim)
            fx0 = func(x0)
            total_calls += 1
            if fx0 < best_fx:
                best_fx = fx0
                best_x = x0.copy()
            C = np.eye(self.dim)
            bx, bfx, c = self._cma_step(func, best_x, 1.5, C, rng, deadline, remaining)
            total_calls += c
            if bfx < best_fx:
                best_fx = bfx
                best_x = bx

        return best_x, best_fx
# EVOLVE-BLOCK-END
