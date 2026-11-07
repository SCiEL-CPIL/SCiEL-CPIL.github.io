"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, deadline):
        """Run CMA-ES from x0 with initial step size sigma0 for up to budget calls."""
        n = self.dim
        x = np.array(x0, dtype=float)
        sigma = sigma0
        fx = func(x)
        calls = 1

        # CMA-ES parameters
        lambda_ = 4 * n  # population size
        mu = lambda_ // 2  # recombination weight
        # weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / w.sum()
        # step size control
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
        # covariance matrix adaptation
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        # state
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        # diagonal of C for efficiency
        D = np.ones(n)
        B = np.eye(n)
        # B D^2 B^T = C, initially identity

        for _ in range(budget - calls):
            if time.time() > deadline:
                break

            # sample population
            Y = np.zeros((lambda_, n))
            for i in range(lambda_):
                Y[i] = B @ (D * rng.standard_normal(n))
            # normalize
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            # generate
            X = np.tile(x, (lambda_, 1)) + sigma * Y
            # evaluate
            F = np.array([func(np.clip(xi, -5, 5)) for xi in X])
            calls += lambda_

            # sort
            order = np.argsort(F)
            X = X[order]
            F = F[order]

            # update mean
            x_new = X[:mu] @ w[:, None]
            # update step size
            y_sigma = (B @ (D @ pc)) / sigma if sigma > 1e-15 else np.zeros(n)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (x_new - x)
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            # update covariance
            pc = (1 - c_c) * pc + c_c * (x_new - x) / sigma
            delta_h = 1 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / (n + 1)) * chi_n else 0
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) - C) + \
                c_mu / mueff * np.sum(w[:, None, None] * X[:mu, :, None] * X[:mu, None, :], axis=0)
            # update B, D via eigendecomposition (every 10th step for efficiency)
            if calls % 10 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                B = eigvecs

            x = x_new
            fx = F[0]

        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = None
        best_fx = np.inf

        # Phase 1: Deterministic grid of key points to establish early progress
        # Evaluate center, origin, and axis-aligned points
        grid_points = [np.zeros(n)]
        # Add midpoints of each axis
        for i in range(n):
            for val in [-5.0, 5.0, -2.5, 2.5]:
                p = np.zeros(n)
                p[i] = val
                grid_points.append(p)
        # Add some random points for diversity
        rng = np.random.default_rng(42)
        for _ in range(10):
            grid_points.append(rng.uniform(-5, 5, n))

        # Evaluate grid points
        for p in grid_points:
            if time.time() > deadline:
                break
            if calls_made >= total_budget:
                break
            p_clipped = np.clip(p, -5, 5)
            fp = func(p_clipped)
            calls_made += 1
            if fp < best_fx:
                best_fx = fp
                best_x = p_clipped.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_fx = func(best_x)
            calls_made = 1

        # Phase 2: Multi-start CMA-ES
        # Use remaining budget for CMA-ES from best point and a few other promising starts
        remaining = total_budget - calls_made
        if remaining <= 0:
            return best_x, best_fx

        # Split remaining budget among a few restarts
        num_restarts = min(3, remaining // 50)  # at least 50 calls per restart
        if num_restarts < 1:
            num_restarts = 1
        calls_per_restart = remaining // num_restarts

        start_points = [best_x.copy()]
        # Add a few other good points from grid
        for p in grid_points:
            if len(start_points) >= num_restarts:
                break
            if not np.allclose(p, best_x, atol=1.0):
                start_points.append(np.clip(p, -5, 5).copy())
        # Fill with random if needed
        while len(start_points) < num_restarts:
            start_points.append(rng.uniform(-5, 5, n))

        for i, sp in enumerate(start_points):
            if time.time() > deadline:
                break
            alloc = calls_per_restart if i < num_restarts - 1 else remaining - calls_made + calls_per_restart
            alloc = min(alloc, total_budget - calls_made)
            if alloc < 10:
                break
            x_res, f_res, n_calls = self._cma_es(func, sp, 1.0, alloc, deadline)
            calls_made += n_calls
            if f_res < best_fx:
                best_fx = f_res
                best_x = x_res.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
