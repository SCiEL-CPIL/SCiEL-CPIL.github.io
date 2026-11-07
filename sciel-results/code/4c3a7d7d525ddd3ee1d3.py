"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with initial step sigma0 for up to budget calls."""
        d = self.dim
        n = max(4, int(4 + 3 * np.log(d)))  # population size
        calls = 0
        x = np.asarray(x0, dtype=float)
        fx = func(x)
        calls += 1
        best_x = x.copy()
        best_f = fx

        sigma = sigma0
        mean = x.copy()
        C = np.eye(d)
        Cinv_sqrt = np.eye(d)  # approximation, will update
        pc = np.zeros(d)
        cs = np.zeros(d)
        # Cumulation parameters
        mu = n // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - d - 2) / (d + mueff)) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        while calls < budget:
            if time.time() > deadline:
                break
            # Generate population
            pops = []
            fits = []
            for i in range(n):
                z = rng.standard_normal(d)
                # Simple approximation: use identity for C^{1/2}
                y = mean + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                pops.append(y)
                fits.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

            if len(pops) < n:
                break

            # Sort by fitness
            order = np.argsort(fits)
            pops = [pops[i] for i in order]
            fits = [fits[i] for i in order]

            # Update mean
            y_w = np.zeros(d)
            for i in range(mu):
                y_w += w[i] * (pops[i] - mean)
            mean = mean + y_w

            # Update step size (simple CSA)
            hsig = (np.sqrt((1 - (1 - c_sigma) ** (budget - calls))) / chi_n) * np.sqrt(
                1 - (1 - c_sigma) ** (2 * (budget - calls))
            ) if budget - calls > 0 else 0
            # Simplified: just use basic adaptation
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(y_w) / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)

            # Update covariance (simplified: rank-mu update)
            y_w_norm = y_w / sigma if sigma > 0 else y_w
            C = (1 - c_1 - c_mu) * C + c_1 * (2 - mu_c) * np.outer(mean - x0, mean - x0) + c_mu * sum(
                w[i] * np.outer(pops[i] - mean, pops[i] - mean) for i in range(mu)
            )
            # Ensure positive definite
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(d)

            # Stop if step size is too small
            if sigma < 1e-8 or np.all(np.abs(y_w) < 1e-10):
                break

        return best_x, best_f, calls

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        used = 0

        # Initial random sampling to get a good starting point
        n_init = min(20, budget)
        init_x = []
        init_f = []
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            used += 1
            init_x.append(x)
            init_f.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if used >= budget or time.time() > deadline:
            return best_x, best_f

        # Run CMA-ES from best initial point
        remaining = budget - used
        # Use a few restarts with different initial points
        n_restarts = min(3, max(1, remaining // 100))
        per_restart = remaining // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if r == 0:
                x0 = best_x.copy()
                sigma0 = 1.0
            else:
                # Pick another good initial point
                idx = np.argmin(init_f)
                x0 = init_x[idx].copy() + rng.normal(0, 0.5, d)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 0.5

            bx, bf, calls = self._cma_es(func, x0, sigma0, per_restart, rng, deadline)
            used += calls
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
            if used >= budget or time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
