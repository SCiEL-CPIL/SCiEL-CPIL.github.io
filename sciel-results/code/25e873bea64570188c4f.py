"""Two-phase hybrid: dense deterministic axis/radial sweep to identify the Discus needle,
then aggressive CMA-ES for fast exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: deterministic sweep + CMA-ES for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            if calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Dense deterministic sweep to find the narrow axis of Discus
        # BBOB f11 (Discus): f(x) = 0.5 * (||x||^2 + 2 * max_i |x_i|^2)
        # The needle is along the axis where the max component is large.
        
        # First, evaluate origin
        evaluate(np.zeros(dim))

        # Axis-based sweep: for each axis, try positive and negative directions at several radii
        radii = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        for axis in range(dim):
            for r in radii:
                if calls >= budget or time.time() > deadline:
                    break
                # Positive direction
                x = np.zeros(dim)
                x[axis] = r
                evaluate(x)
                # Negative direction
                x[axis] = -r
                if calls < budget and time.time() < deadline:
                    evaluate(x)
            if calls >= budget or time.time() > deadline:
                break

        # Radial sweep in random directions
        n_radial = 20
        for i in range(n_radial):
            if calls >= budget or time.time() > deadline:
                break
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
                if calls >= budget or time.time() > deadline:
                    break
                evaluate(direction * r)

        # Phase 2: CMA-ES with aggressive settings for fast convergence
        if calls < budget:
            x0 = best_x.copy() if best_x is not None else np.zeros(dim)
            self._cma_es(func, x0, budget - calls, deadline, rng, evaluate)

        return best_x, best_f

    def _cma_es(self, func, x0, max_evals, deadline, rng, evaluate):
        """Run CMA-ES starting from x0."""
        dim = self.dim
        n = dim
        
        # CMA-ES parameters
        lambda_ = max(10, 4 + int(3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_c = min(c_c, 1.0)
        mueff_inv = 1.0 / mueff
        c_1 = (2 - mueff_inv + c_sigma) / ((n + 1.3) ** 2 + mueff_inv + c_sigma)
        c_mu = (2 * (mueff_inv - 1) + c_sigma) / ((n + 2) ** 2 + mueff_inv + c_sigma)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff_inv - 1) / (n + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        B = np.eye(n)
        D = np.ones(n)
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x_mean = x0.copy()
        sigma = 1.5  # Larger initial sigma for faster adaptation to anisotropy
        evals = 0

        # Initial population
        pop_x = []
        pop_f = []
        for i in range(lambda_):
            z = rng.standard_normal(n)
            xi = x_mean + sigma * (B @ (D * z))
            fi = evaluate(xi)
            if np.isinf(fi):
                break
            pop_x.append(xi)
            pop_f.append(fi)
            evals += 1
            if evals >= max_evals:
                break

        if len(pop_x) < 2:
            return

        # Sort by fitness
        idx = np.argsort(pop_f)
        pop_x = [pop_x[i] for i in idx]
        pop_f = [pop_f[i] for i in idx]

        while evals < max_evals and time.time() < deadline:
            # Update translation (cm)
            w = weights[:min(mu, len(pop_f))]
            x_mean_new = np.sum(np.array(pop_x[:len(w)]) * w[:, None], axis=0)

            pc = (1 - c_c) * pc + c_c * np.sqrt(mueff) / (sigma * np.sqrt(np.sum(D ** 2))) * (x_mean_new - x_mean)
            hs = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (evals + 1))) / chi_n < 1.4 + 2 / (n + 1)
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hs) * c_c * (2 - c_c) * C) + \
                c_mu * np.sum(w[:, None, None] * (np.array(pop_x[:len(w)]) - x_mean) @ (np.array(pop_x[:len(w)]) - x_mean).T, axis=0)

            # Eigendecomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-10))
            B = B * np.sqrt(D[:, None])

            # Update sigma (cs)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / (sigma * np.sqrt(np.sum(D ** 2))) * (x_mean_new - x_mean)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 1e6)
            sigma = max(sigma, 1e-12)

            x_mean = x_mean_new

            # Generate new population
            pop_x_new = []
            pop_f_new = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                xi = x_mean + sigma * (B @ z)
                fi = evaluate(xi)
                if np.isinf(fi):
                    break
                pop_x_new.append(xi)
                pop_f_new.append(fi)
                evals += 1
                if evals >= max_evals:
                    break

            if len(pop_x_new) < 2:
                break

            idx_new = np.argsort(pop_f_new)
            pop_x = [pop_x_new[i] for i in idx_new]
            pop_f = [pop_f_new[i] for i in idx_new]
# EVOLVE-BLOCK-END
