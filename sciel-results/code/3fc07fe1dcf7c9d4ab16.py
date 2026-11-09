"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive sigma and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, sigma0, budget, deadline, rng):
        """Run CMA-ES from x0 with given budget, respecting deadline."""
        dim = self.dim
        n = dim
        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        # Weights
        weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        x = np.array(x0, dtype=float)
        sigma = sigma0
        ps = np.zeros(n)
        pc = np.zeros(n)

        def _phi(k):
            return np.sqrt((n + 1) * (2 - 1 / (n + 1))) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial Hessian approximation (identity)
        hessian = np.eye(n)
        hessian_eigvals = np.ones(n)
        hessian_eigvecs = np.eye(n)

        best_x = x.copy()
        best_f = func(x)
        calls = 1

        for gen in range(budget // lam + 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                # Standard normal
                z = rng.standard_normal(n)
                # Transform using current Hessian approximation
                y = x + sigma * (hessian_eigvecs @ (np.sqrt(hessian_eigvals) * (hessian_eigvecs.T @ z)))
                y = np.clip(y, -5, 5)
                offspring.append(y)
                calls += 1

            if not offspring:
                break

            # Evaluate offspring
            offspring_f = np.array([func(y) for y in offspring])
            calls += len(offspring)

            # Sort by fitness
            idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in idx]
            offspring_f = offspring_f[idx]

            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()

            # Update mean
            w = weights[:mueff]
            x_new = sum(w[i] * offspring[i] for i in range(int(mueff)))

            # Update Hessian (covariance) approximation
            y_inv_sigma = [(offspring[i] - x) / sigma for i in range(int(mueff))]
            C_new = (1 - c_1 - c_mu) * hessian
            for i in range(int(mueff)):
                C_new += w[i] * np.outer(y_inv_sigma[i], y_inv_sigma[i])
            # Add (1-c1-cmu) * old C + c1 * (pc pc^T + (1-delta_sigma) pc^T pc)
            C_new += c_1 * (np.outer(pc, pc) + (1 - np.dot(pc, pc)) * np.eye(n))
            # Ensure symmetry and positive semi-definite
            C_new = (C_new + C_new.T) / 2
            # Clip small eigenvalues
            eigvals, eigvecs = np.linalg.eigh(C_new)
            eigvals = np.maximum(eigvals, 1e-12)
            C_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
            hessian = C_new
            hessian_eigvals = eigvals
            hessian_eigvecs = eigvecs

            # Update path for covariance
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (hessian_eigvecs.T @ ((x_new - x) / sigma))
            hs = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam + 1))) < (1.4 + 2 / (n + 1)) * np.trace(np.eye(n)) / n else 0.0
            pc = (1 - c_c) * pc + c_c * hs * (hessian_eigvecs.T @ ((x_new - x) / sigma))

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / _phi(1) - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            x = x_new

            if calls >= budget:
                break

        return best_x, best_f

    def __call__(self, func):
        deadline = time.time() + 280  # Safety margin before 300s
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Latin Hypercube initialization for diverse starting points
        n_starts = 10
        lhc_points = []
        for _ in range(n_starts):
            # LHS: divide each dimension into n_starts strata, pick one random point per stratum
            pts = []
            for d in range(dim):
                stratum = rng.integers(0, n_starts)
                lo = -5 + 10 * stratum / n_starts
                hi = -5 + 10 * (stratum + 1) / n_starts
                pts.append(rng.uniform(lo, hi))
            lhc_points.append(np.array(pts))

        # Evaluate all starting points
        start_vals = [(pt, func(pt)) for pt in lhc_points]
        start_vals.sort(key=lambda t: t[1])

        best_x = start_vals[0][0].copy()
        best_f = start_vals[0][1]
        calls_made = n_starts

        # Allocate remaining budget among top starts
        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Give more budget to better starts
        # Top 3 starts get 40%, 30%, 20% of remaining; rest get 10%
        fracs = [0.4, 0.3, 0.2, 0.1]
        while len(fracs) < n_starts:
            fracs.append(0.0)
        fracs = fracs[:n_starts]

        for i in range(n_starts):
            if time.time() > deadline:
                break
            b_i = int(remaining * fracs[i])
            if b_i < 10:
                b_i = 10 if remaining >= 10 else 0
            if b_i == 0:
                break
            x0 = start_vals[i][0]
            f0 = start_vals[i][1]
            # Initial sigma based on distance from best
            sigma0 = 1.0
            bx, bf = self._cma_es(func, x0, sigma0, b_i, deadline, rng)
            if bf < best_f:
                best_f = bf
                best_x = bx
            calls_made += b_i
            remaining -= b_i
            if remaining <= 0:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
