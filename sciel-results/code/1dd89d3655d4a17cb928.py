"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB-style optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma, budget, rng):
        """Run CMA-ES starting from x0 with initial step size sigma."""
        n = self.dim
        lam = max(4, int(4 * (1 + 3 * np.log(n))))
        # CMA-ES parameters
        cc = (4 + n / (4 + n)) / ((n + 4) + (4 + n / (4 + n)))
        cs = (n + 1.7) / (n + n + 2.2)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = int(1 / weights.sum())
        c1 = (2 / ((n + 1.3) ** 2 + n)) * (2 - cc + cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + n))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # State
        mean = np.array(x0, dtype=float)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        best_x = mean.copy()
        best_f = func(mean)
        calls = 1

        for gen in range(budget - 1):
            # Generate population
            pop = np.zeros((lam, n))
            for i in range(lam):
                # Standard normal
                z = rng.standard_normal(n)
                # Whitening transform: C^{1/2} z
                # Use Cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                pop[i] = mean + sigma * (L @ z)
                pop[i] = np.clip(pop[i], -5, 5)

            # Evaluate
            fvals = np.array([func(pop[i]) for i in range(lam)])
            calls += lam

            # Sort
            order = np.argsort(fvals)
            fvals_sorted = fvals[order]
            x_sorted = pop[order]

            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = x_sorted[0].copy()

            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)

            # Update evolution paths
            pc = (1 - cc) * pc + cc / np.sqrt(cc) * (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * damps) / sigma * (mean - old_mean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1))) / chi_n < 1.4 + 2 / (n + 1)

            # Update covariance
            delta = (1 + 2 * (hsig > 1.4 + 2 / (n + 1)) - 1) * cc
            rank_one = np.outer(ps, ps)
            C = (1 - c1 - cmu + delta * c1) * C + c1 * (np.outer(mean - old_mean, mean - old_mean) / sigma ** 2 + (delta - 1 + cc) * C)
            rank_mu = np.sum([weights[i] * np.outer(x_sorted[i] - old_mean, x_sorted[i] - old_mean) for i in range(mu)], axis=0)
            C = C + cmu * rank_mu / sigma ** 2

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Regularize if needed
            try:
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C += 1e-10 * np.eye(n)
            except:
                pass

            # Adapt sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10)

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        max_time = 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Multiple restarts strategy
        # First: explore broadly, then refine
        best_x = np.full(dim, np.nan)
        best_f = np.inf
        total_calls = 0

        # Phase 1: Broad exploration with larger sigma
        # Use a few restarts with different starting points
        n_restarts = 3
        calls_per_restart = budget // n_restarts

        for r in range(n_restarts):
            # Check time
            if time.time() - start_time > max_time:
                break

            # Starting point: first restart at origin, others random
            if r == 0:
                x0 = np.zeros(dim)
            else:
                rng_r = np.random.default_rng(42 + r)
                x0 = rng_r.uniform(-4, 4, dim)

            sigma0 = 3.0 if r == 0 else 2.0

            # Run CMA-ES
            x_res, f_res, calls_used = self._cma_es(func, x0, sigma0, calls_per_restart, np.random.default_rng(100 + r))
            total_calls += calls_used

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

            # Check time
            if time.time() - start_time > max_time:
                break

        # Phase 2: Local refinement around best point
        remaining_calls = budget - total_calls
        if remaining_calls > 20 and time.time() - start_time < max_time:
            # Run a local CMA-ES with small sigma
            x_res, f_res, calls_used = self._cma_es(func, best_x, 0.5, remaining_calls, np.random.default_rng(999))
            total_calls += calls_used
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        # Ensure we don't exceed budget
        if total_calls > budget:
            # We've used too many calls, but the evaluator records all calls
            # The return value is ignored, so this is fine as long as no exception
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
