"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with adaptive step size and covariance matrix."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Initialize with center of box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        # Adaptation parameters
        mu = 4  # number of parents
        lambda_ = 8  # population size
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Precompute cholesky
        def _update_chol():
            nonlocal C, C_inv, C_sqrt, C_inv_sqrt
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Symmetrize
            C = 0.5 * (C + C.T)
            C_inv = np.linalg.inv(C)
            C_inv = 0.5 * (C_inv + C_inv.T)
            C_sqrt = np.linalg.cholesky(C)
            C_inv_sqrt = np.linalg.cholesky(C_inv)
            # Make C_inv_sqrt upper triangular then transpose to get lower
            # Actually cholesky gives lower triangular L such that L L^T = A
            # We want B such that B B^T = C_inv, so B = L
            C_inv_sqrt = C_inv_sqrt

        _update_chol()

        calls_made = 1

        for gen in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate population
            ys = np.zeros((lambda_, dim))
            fvals = np.zeros(lambda_)

            for i in range(lambda_):
                z = np.random.randn(dim)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                ys[i] = y
                fvals[i] = func(y)
                calls_made += 1

                if fvals[i] < best_fx:
                    best_fx = fvals[i]
                    best_x = y.copy()

                if time.time() > deadline:
                    break

            if calls_made >= budget:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            ys = ys[order]
            fvals = fvals[order]

            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * ys[i]
            x = x_new

            # Update step size control variable
            hsig = (1 + 2 * min(1, (mueff - 1) / (dim + 1)), 1)  # simplified
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (C_inv_sqrt.T @ (x - x_new) / sigma)
            # Actually the standard formula is:
            # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mueff) * C_old^{-1/2} * (x_new - x_old)/sigma
            # Let me redo this properly

            # Recompute with proper update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ ((x_new - (x_new - (x_new - x))) / sigma))
            # This is getting complicated. Let me use a simpler but effective approach.

            # Simpler CMA-ES implementation
            break

        return best_x, best_fx

    def _simple_cma_es(self, func):
        """A simpler, more robust CMA-ES variant."""
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280

        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        sigma = 1.5
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Population and selection
        lam = 10
        mu = 5
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Cholesky decomposition
        def _chol_update():
            nonlocal C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = (eigvecs * eigvals) @ eigvecs.T
            C = 0.5 * (C + C.T)
            return np.linalg.cholesky(C)

        C_sqrt = _chol_update()
        C_inv_sqrt = np.linalg.inv(C_sqrt)

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate sample points
            samples = []
            fvals = []
            for _ in range(lam):
                z = np.random.randn(dim)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                samples.append(y)
                fvals.append(fy)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                if calls >= budget or time.time() > deadline:
                    break

            if len(samples) < mu:
                break

            # Sort and select
            combined = sorted(zip(fvals, samples), key=lambda t: t[0])
            selected = [s for _, s in combined[:mu]]

            # New mean
            x_new = np.zeros(dim)
            for i, s in enumerate(selected):
                x_new += weights[i] * s
            x_new = np.clip(x_new, -5, 5)

            # Update covariance
            diff = np.array([s - x for s in selected])
            C = (1 - (c_1 + c_mu)) * C + c_1 * (np.outer(p_c, p_c) + np.eye(dim) - C) + c_mu * (diff.T @ np.diag(weights) @ diff)
            C = 0.5 * (C + C.T)

            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * C_inv_sqrt @ (x_new - x) / sigma

            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (x_new - x) / sigma

            # Adapt sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            x = x_new

            # Update Cholesky
            C_sqrt = _chol_update()
            C_inv_sqrt = np.linalg.inv(C_sqrt)

        return best_x, best_fx


class Optimizer:
    """CMA-ES based optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 275  # safety margin

        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # CMA-ES parameters
        sigma = 1.5
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        lam = 10  # population size
        mu = 5   # number of parents
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        def _update_chol():
            nonlocal C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = (eigvecs * eigvals) @ eigvecs.T
            C = 0.5 * (C + C.T)
            return np.linalg.cholesky(C)

        C_sqrt = _update_chol()
        C_inv_sqrt = np.linalg.inv(C_sqrt)

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            samples = []
            fvals = []
            for _ in range(lam):
                if time.time() > deadline or calls >= budget:
                    break
                z = np.random.randn(dim)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                samples.append(y)
                fvals.append(fy)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            if len(samples) < mu:
                break

            # Sort and select
            indices = np.argsort(fvals)
            selected = [samples[i] for i in indices[:mu]]

            # New mean
            x_new = np.zeros(dim)
            for i, s in enumerate(selected):
                x_new += weights[i] * s
            x_new = np.clip(x_new, -5, 5)

            # Update covariance matrix
            diff = np.array([s - x for s in selected])
            rank_one = (1 - c_1 - c_mu) * C
            rank_one += c_1 * (np.outer(p_c, p_c) + np.eye(dim) - C)
            rank_mu = c_mu * (diff.T @ np.diag(weights) @ diff)
            C = rank_one + rank_mu
            C = 0.5 * (C + C.T)

            # Update evolution path p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * C_inv_sqrt @ (x_new - x) / sigma

            # Update evolution path p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (x_new - x) / sigma

            # Adapt sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(c_sigma / c_sigma * min(1, (norm_p_sigma / chi_n - 1) / dim))
            sigma = np.clip(sigma, 1e-10, 100)

            x = x_new

            # Update Cholesky decomposition
            C_sqrt = _update_chol()
            C_inv_sqrt = np.linalg.inv(C_sqrt)

        return best_x, best_fx
# EVOLVE-BLOCK-END
