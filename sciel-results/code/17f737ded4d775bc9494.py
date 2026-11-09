"""Rewritten with a proper CMA-ES optimizer for the Bent Cigar function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    CMA-ES adapts both the step size and the full covariance matrix, making it
    highly effective on anisotropic functions like the Bent Cigar (f12), where
    one dimension is very flat and the others are steep.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, deadline, rng, best_x_ref, best_fx_ref):
        """Run CMA-ES from x0 with initial sigma. Returns best_x, best_fx."""
        dim = self.dim
        n = dim
        # Population size: standard recommendation
        lam = int(4 + int(3 * np.log(n)))  # for n=5, lam = 4 + 3*1.6 = 8 -> use 14 for robustness
        lam = max(8, int(4 + 3 * np.log(n)))
        mu = lam // 2

        # Learning rates
        mu_w = np.arange(1, mu + 1)
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

        # State
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        C_inv = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        best_x = x.copy()
        best_fx = func(x)

        # Precompute Cholesky of C (identity initially)
        C_chol = np.linalg.cholesky(C)

        calls = 1
        gen = 0

        while calls < budget and time.time() < deadline:
            # Generate population
            ys = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x + sigma * (C_chol @ z)
                ys.append(y)
            Y = np.array(ys)

            # Evaluate
            fvals = np.empty(lam)
            for i in range(lam):
                if calls >= budget or time.time() >= deadline:
                    break
                fvals[i] = func(Y[i])
                calls += 1

            # If we ran out, pad with worst
            if calls >= budget or time.time() >= deadline:
                if not np.all(np.isfinite(fvals[:calls - 1])):
                    pass
                # Only use the ones we evaluated
                n_eval = calls - 1
                if n_eval < mu:
                    break
                fvals = fvals[:n_eval]
                Y = Y[:n_eval]
                # Sort and take best mu
                order = np.argsort(fvals)
                # Update best
                if fvals[order[0]] < best_fx:
                    best_fx = fvals[order[0]]
                    best_x = Y[order[0]].copy()
                break

            # Sort by fitness
            order = np.argsort(fvals)
            Y_sorted = Y[order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_fx:
                best_fx = fvals_sorted[0]
                best_x = Y_sorted[0].copy()

            # Weighted mean of best mu
            y_mean = np.sum(w[:, None] * Y_sorted[:mu], axis=0)

            # Update covariance step size control vector ps
            C_sqrt = C_chol  # lower triangular, so C = C_chol @ C_chol.T
            # ps = ps * (1 - cs) + sqrt(cs*(2-cs)*mueff) / sigma * (C_sqrt^-1 @ (y_mean - x))
            C_sqrt_inv = np.linalg.inv(C_sqrt)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_sqrt_inv @ (y_mean - x))

            # Update evolution path pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))) / chi_n < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (y_mean - x) / sigma

            # Update mean
            x = x + y_mean

            # Update covariance matrix
            # Delta h: correction for the hsig
            delta_h = (1 - hsig) * cc * (2 - cc)
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(w[:, None, None] * np.outer(Y_sorted[:mu] - x, Y_sorted[:mu] - x), axis=0)
            C = (1 + cs * delta_h - cs - c1 - cmu) * C + cs * rank_one + c1 * np.eye(n) + cmu * rank_mu

            # Symmetrize and ensure positive definiteness
            C = (C + C.T) / 2

            # Update sigma
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Recompute Cholesky
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small diagonal to make it PD
                C = C + 1e-6 * np.eye(n)
                C = (C + C.T) / 2
                C_chol = np.linalg.cholesky(C)

            gen += 1

        return best_x, best_fx

    def __call__(self, func):
        deadline = time.time() + 275  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        dim = self.dim

        # Track global best
        best_x = np.zeros(dim)
        best_fx = float('inf')

        # Initial evaluation at origin
        fx0 = func(np.zeros(dim))
        best_x = np.zeros(dim).copy()
        best_fx = fx0
        calls_used = 1

        # Multi-start CMA-ES: try a few different starting points and sigma values
        # The Bent Cigar with hidden shift/rotation means we need to be robust
        # We'll do a few restarts with different initializations

        # First, do a quick random sampling to get a sense of the landscape
        # and find a good starting point
        n_samples = min(20, self.budget // 10)
        for i in range(n_samples):
            if calls_used >= self.budget or time.time() >= deadline:
                break
            x_try = rng.uniform(-5, 5, dim)
            fx_try = func(x_try)
            calls_used += 1
            if fx_try < best_fx:
                best_fx = fx_try
                best_x = x_try.copy()

        # Now run CMA-ES from the best point found
        # Use a moderate initial sigma
        sigma0 = 1.5

        # Run CMA-ES with the remaining budget
        remaining_budget = self.budget - calls_used
        if remaining_budget > 10:
            bx, bfx = self._cma_es(func, best_x, sigma0, calls_used + remaining_budget,
                                    deadline, rng, best_x, best_fx)
            if bfx < best_fx:
                best_fx = bfx
                best_x = bx.copy()
            calls_used = calls_used + remaining_budget

        # If we still have budget, do another restart from a slightly perturbed best
        if calls_used < self.budget and time.time() < deadline:
            # Perturb the best solution
            x_restart = best_x + rng.standard_normal(dim) * 0.5
            x_restart = np.clip(x_restart, -5, 5)
            remaining = self.budget - calls_used
            if remaining > 10:
                bx, bfx = self._cma_es(func, x_restart, 0.5, calls_used + remaining,
                                        deadline, rng, best_x, best_fx)
                if bfx < best_fx:
                    best_fx = bfx
                    best_x = bx.copy()
                calls_used = calls_used + remaining

        return best_x, best_fx
# EVOLVE-BLOCK-END
