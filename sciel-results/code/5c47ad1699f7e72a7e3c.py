"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with L-BFGS-B refinement for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: CMA-ES for global exploration ---
        # Bent cigar is ill-conditioned: f(x) = 100*(x1^2 + sum_{i=2}^d 10^(i-2)*xi^2)
        # CMA-ES adapts covariance to handle this.

        # Initialize
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        lambda_ = 4 * (int(np.log(dim)) + 1)  # population size
        sigma = 2.0  # initial step size
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 4)
        c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        cmu = min(0.4 - c1, 2 / ((dim + 2) ** 2 + lambda_))
        cmu = max(cmu, 0)
        cc = c_c
        cs = c_sigma
        damps = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (dim + 1)) - 1) + c_sigma
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Precompute for efficiency
        def mat_sqrt_inv(mat):
            """Compute inverse square root of a symmetric positive definite matrix."""
            eigvals, eigvecs = np.linalg.eigh(mat)
            eigvals = np.clip(eigvals, 1e-15, None)
            inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return inv_sqrt

        def mat_sqrt(mat):
            """Compute square root of a symmetric positive definite matrix."""
            eigvals, eigvecs = np.linalg.eigh(mat)
            eigvals = np.clip(eigvals, 1e-15, None)
            sqrt_mat = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            return sqrt_mat

        C_sqrt = mat_sqrt(C)

        calls_used = 1
        no_improve = 0
        max_no_improve = 150  # switch to L-BFGS-B after this many non-improving generations

        for gen in range(1, 200):  # max 200 generations
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Generate population
            z = rng.standard_normal((lambda_, dim))
            pop = np.array([x + sigma * (C_sqrt @ z[i]) for i in range(lambda_)])
            pop = np.clip(pop, -5, 5)

            # Evaluate
            fvals = np.array([func(pop[i]) for i in range(lambda_)])
            calls_used += lambda_

            # Sort by fitness
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]

            # Update best
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = pop[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Update mean
            w = np.log(np.arange(1, lambda_ + 1)) - np.log(np.arange(1, lambda_ + 1)).mean()
            w = w / w.sum()
            mu_eff = 1.0 / (w ** 2).sum()

            x_old = x.copy()
            x = (w[:lambda_] * pop).sum(axis=0)

            # Update evolution paths
            delta = (x - x_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt @ delta)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) <
                    (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * delta

            # Update covariance
            C = ((1 - c1 - cmu) * C +
                 c1 * (np.outer(pc, pc) + cc * (2 - cc) * C) +
                 cmu / mu_eff * (w[:lambda_] * np.outer(pop - x_old, pop - x_old)).sum(axis=0))

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update C_sqrt
            C_sqrt = mat_sqrt(C)

            # Adapt sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-6, 50.0)

            if no_improve > max_no_improve:
                break

        # --- Phase 2: L-BFGS-B refinement ---
        if time.time() < deadline and calls_used < budget:
            # Use the best point found by CMA-ES as starting point
            x0 = best_x.copy()

            def objective(x):
                return func(np.clip(x, -5, 5))

            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
