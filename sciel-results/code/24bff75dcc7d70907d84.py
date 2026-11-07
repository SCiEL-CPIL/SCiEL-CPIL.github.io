"""Rewritten optimizer for BBOB-style Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: coarse grid, CMA-ES, local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 280  # safety margin before 300s limit

    def __call__(self, func):
        self.calls = 0
        self.max_calls = self.budget
        self.best_x = None
        self.best_f = np.inf

        def _eval(x):
            """Evaluate with call counting and deadline check."""
            nonlocal calls, best_x, best_f
            if self.calls >= self.max_calls:
                return None
            if time.time() > self.deadline:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            self.calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid search to find promising region
        # Use a small number of well-spaced points
        n_coarse = 30
        if n_coarse < self.max_calls:
            # Create a stratified sample
            rng = np.random.default_rng(42)
            # Start with center and some structured points
            points = []
            # Center
            points.append(np.zeros(self.dim))
            # Axis-aligned points at various scales
            for scale in [1.0, 2.0, 3.0, 4.0]:
                for axis in range(self.dim):
                    for sign in [-1, 1]:
                        p = np.zeros(self.dim)
                        p[axis] = sign * scale
                        points.append(p)
            # Random points
            while len(points) < n_coarse:
                points.append(rng.uniform(-5, 5, self.dim))

            for p in points:
                if self.calls >= self.max_calls or time.time() > self.deadline:
                    break
                _eval(p)

        if self.best_x is None:
            # Fallback: just do a simple (1+1)-ES
            x = np.zeros(self.dim)
            fx = _eval(x)
            sigma = 2.0
            rng = np.random.default_rng(0)
            for _ in range(self.max_calls - 1):
                if time.time() > self.deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = _eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
            return self.best_x, self.best_f

        # Phase 2: CMA-ES starting from best point found
        x0 = self.best_x.copy()
        sigma0 = 1.5  # Initial step size
        n = self.dim
        popsize = 4 + int(3 * np.log(n))  # Standard CMA-ES population size
        popsize = max(popsize, 5)

        # CMA-ES parameters
        lambda_ = popsize
        mu = lambda_ // 2  # Number of weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 - cc + min(mu / (mueff + 2), 2 / (n + 1))) / ((n + 1.3) ** 2 + cc)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + cc))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # Initialize
        x_mean = x0.copy()
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        sigma = sigma0
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Budget remaining for CMA-ES
        budget_left = self.max_calls - self.calls
        # Reserve some calls for local refinement
        reserve = min(50, budget_left // 5)
        cma_budget = budget_left - reserve

        generations = cma_budget // lambda_
        if generations < 1:
            generations = 1

        for gen in range(generations):
            if self.calls >= self.max_calls or time.time() > self.deadline:
                break

            # Generate population
            xs = []
            for i in range(lambda_):
                # Sample from N(x_mean, sigma^2 * C)
                # Use Cholesky-like decomposition via eigendecomposition
                # For efficiency, use a simple approach
                z = np.random.default_rng(42 + gen * 100 + i).standard_normal(n)
                # Transform: C^{1/2} z
                # Compute C^{1/2} via eigenvalue decomposition (cache would be better)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                xi = x_mean + sigma * (C_sqrt @ z)
                xs.append(np.clip(xi, -5, 5))

            # Evaluate
            fs = []
            for xi in xs:
                fi = _eval(xi)
                if fi is None:
                    fi = np.inf
                fs.append(fi)
            fs = np.array(fs)

            # Sort by fitness
            idx = np.argsort(fs)
            xs_sorted = np.array([xs[i] for i in idx])
            fs_sorted = fs[idx]

            # Update best
            if fs_sorted[0] < self.best_f:
                self.best_f = fs_sorted[0]
                self.best_x = xs_sorted[0].copy()

            # Update mean
            x_mean_new = x_mean + sigma * sum(weights[i] * (xs_sorted[i] - x_mean) for i in range(mu))
            x_mean = np.clip(x_mean_new, -5, 5)

            # Update evolution path
            # C^{-1/2} computation
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            y_mean = (x_mean_new - x_mean) / sigma if sigma > 1e-15 else np.zeros(n)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ y_mean
            pc = (1 - cc) * pc + cc / sigma * (x_mean_new - x_mean)

            # Update covariance
            delta_sig = 1.0 + (1 - cc) * (1 - np.dot(pc, pc) / (n * (1 - (1 - cc) ** 2)))
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                vi = C_inv_sqrt @ (xs_sorted[i] - x_mean)
                rank_mu += weights[i] * np.outer(vi, vi)

            C = (1 + c1 * delta_sig + cmu - c1 - cmu) * C + c1 * np.outer(ps, ps) + cmu * rank_mu

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Check for convergence
            if sigma < 1e-8 or np.linalg.norm(C) < 1e-10:
                break

        # Phase 3: Local refinement with smaller steps
        if self.calls < self.max_calls and time.time() < self.deadline:
            x = self.best_x.copy()
            fx = self.best_f
            sigma_local = 0.3
            rng = np.random.default_rng(123)
            local_budget = self.max_calls - self.calls
            for _ in range(local_budget):
                if time.time() > self.deadline:
                    break
                y = np.clip(x + sigma_local * rng.standard_normal(self.dim), -5, 5)
                fy = _eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.1
                else:
                    sigma_local *= 0.9
                    if sigma_local < 0.01:
                        break

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
