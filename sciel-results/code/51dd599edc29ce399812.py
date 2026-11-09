"""CMA-ES optimizer for the BBOB f11 (Discus) task."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Compact CMA-ES with covariance adaptation and CSA step-size control."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin under the 330s limit

        # Population size (mu = lambda/2)
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2

        # Start at the box center (minimizer is guaranteed to lie in [-4,4]^5)
        xmean = np.zeros(dim)
        sigma = 2.0

        # Covariance matrix
        C = np.eye(dim)
        Cinv = np.eye(dim)
        Cinv_sqrt = np.eye(dim)

        # Step-size adaptation (CSA) parameters
        mu_eff = float(mu)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0.0, (mu_eff - 1) / (dim + 1)) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Covariance adaptation parameters
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1,
                   2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        c_1 += c_mu
        c_1 = min(c_1, 2 * c_c)
        c_mu = c_1 - c_1 * c_1 / (2 * c_c)
        chi_1 = np.sqrt(dim) * (1 + 1 / (2 * dim) + 1 / (12 * dim ** 2))
        chi_mu = np.sqrt(2 * mu_eff * (1 - 1 / mu_eff))

        # State
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        def sample_and_eval():
            nonlocal xmean, sigma, C, Cinv, Cinv_sqrt, ps, pc
            ys = np.empty((lam, dim))
            xs = np.empty((lam, dim))
            fs = np.empty(lam)
            for i in range(lam):
                z = np.linalg.solve(Cinv_sqrt, rng.standard_normal(dim))
                xs[i] = np.clip(xmean + sigma * z, -5.0, 5.0)
                fs[i] = func(xs[i])
                ys[i] = (xs[i] - xmean) / sigma if sigma > 0 else z
            # Select best mu
            idx = np.argsort(fs)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_w = (w ** 2).sum()
            yw = (w[:, None] * ys[idx[:mu]]).sum(axis=0)
            new_xmean = np.clip(xmean + sigma * yw, -5.0, 5.0)

            # Step-size adaptation (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) * np.dot(Cinv_sqrt, yw)
            cs_factor = (np.linalg.norm(ps) / chi_n - 1)
            sigma *= np.exp(c_sigma / d_sigma * cs_factor)
            sigma = float(np.clip(sigma, 1e-12, 1e3))

            # Covariance adaptation
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_w) * yw
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget_eval + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            rank1 = np.outer(pc, pc)
            if hsig:
                rank1 += (c_1 / c_c) * np.outer(ps, ps)
            C = (1 + c_c * (c_mu - c_c - mu_w)) * C + c_1 * rank1 + c_mu * np.outer(yw, yw)
            # Numerical safeguard: renormalize trace to dim
            tr = np.trace(C)
            if tr > 0:
                C = C * (dim / tr)

            # Refresh Cinv_sqrt via eigendecomposition (robust, dim=5)
            try:
                eigval, eigvec = np.linalg.eigh(C)
                eigval = np.maximum(eigval, 1e-12)
                Cinv_sqrt = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
                Cinv = eigvec @ np.diag(1.0 / eigval) @ eigvec.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                Cinv = np.eye(dim)
                Cinv_sqrt = np.eye(dim)

            xmean = new_xmean
            best_i = idx[0]
            return xs[best_i], fs[best_i]

        rng = np.random.default_rng(0)
        budget_eval = 0

        # Initial evaluation of the starting point
        xbest = xmean.copy()
        fbest = func(xbest)
        budget_eval += 1

        # Main loop
        while budget_eval < budget:
            if time.time() > deadline:
                break
            # Check if a full generation fits in the budget
            if budget_eval + lam > budget:
                break
            try:
                x, f = sample_and_eval()
            except Exception:
                break
            budget_eval += lam
            if f < fbest:
                fbest = f
                xbest = x.copy()

        return xbest, fbest
# EVOLVE-BLOCK-END
