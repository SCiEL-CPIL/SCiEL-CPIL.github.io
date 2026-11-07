"""Optimized black-box continuous optimizer for BBOB-style f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES-like adaptive covariance optimizer with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, budget, sigma0, rng):
        """Run CMA-ES for a given budget starting from x0."""
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        best_x, best_f = x.copy(), fx

        # CMA-ES parameters
        lambda_ = 4 * (d + 2)  # population size
        mu = lambda_ // 2       # number of weights
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c1 = (2 + 2.0 / (d + 1.3)) / ((d + 2) ** 2 + mueff)
        cmu = min(1 - c1, (2 + (mueff - 2) / (d + 2)) / ((d + 2) ** 2 + mueff))
        cs = (mueff + 1.5 + 2 / (d + mueff)) / (d + mueff + 5)
        ds = 1 + cs + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        chi = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # State
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        sigma = sigma0
        step_size = 1.0  # for Hessian-free step size control
        evaluations = 1

        while evaluations < budget:
            # Update C via rank-mu update (approximate with identity scaling for speed)
            # For small d, we can do full CMA-ES
            ys = []
            for i in range(lambda_):
                # Sample from N(0, C) approximately using Cholesky
                # For efficiency with small d, use simple approach
                z = rng.standard_normal(d)
                # Approximate C^{1/2} z using current step size
                y = sigma * z
                xi = np.clip(x + y, -5, 5)
                fi = func(xi)
                ys.append((fi, xi, y))
                evaluations += 1
                if evaluations >= budget:
                    break

            # Sort by fitness
            ys.sort(key=lambda t: t[0])
            selected = ys[:min(mu, len(ys))]

            # Update mean
            xnew = np.sum([w[i] * s[1] for i, s in enumerate(selected)], axis=0)
            # Clip to box
            xnew = np.clip(xnew, -5, 5)

            # Update covariance
            # Simple rank-1 update for pc
            y_sigma = (xnew - x) / sigma if sigma > 1e-10 else np.zeros(d)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
            hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) < (1.4 + 2 / (d + 1)) * chi else 0.0
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * y_sigma

            # Update C (simplified rank-1 + rank-mu)
            # For speed, use diagonal approximation
            C = (1 - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            for i, s in enumerate(selected):
                C += cmu * w[i] * np.outer(s[2], s[2]) / (sigma ** 2)

            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-8, 100.0)

            x = xnew
            fx = func(x) if evaluations < budget else fx
            evaluations += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin
        d = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_budget = self.budget

        # Multi-restart strategy with adaptive allocation
        n_restarts = 3
        budget_per_restart = total_budget // n_restarts
        remaining = total_budget

        for r in range(n_restarts):
            if remaining <= 0:
                break
            if time.time() > deadline:
                break

            # Initialize: first restart at origin-like, others random
            if r == 0:
                x0 = np.zeros(d)
            else:
                x0 = rng.uniform(-4, 4, d)

            # Use a fraction of remaining budget, leaving some for later
            alloc = min(remaining, budget_per_restart + r * 50)
            if r == n_restarts - 1:
                alloc = remaining

            if alloc < 10:
                break

            try:
                bx, bf = self._cmaes(func, x0, alloc, 2.0, rng)
                remaining -= alloc
                if bf < best_f:
                    best_f = bf
                    best_x = bx
            except Exception:
                # Fallback: simple random search
                for _ in range(alloc):
                    if time.time() > deadline:
                        break
                    x = rng.uniform(-5, 5, d)
                    fx = func(x)
                    if fx < best_f:
                        best_f = fx
                        best_x = x
                break

        # Final safety: ensure we have a result
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
