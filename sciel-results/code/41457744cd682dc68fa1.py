"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Two-stage optimizer: coarse LHS bracketing followed by CMA-ES + BFGS polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        n = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)

        # Stage 1: Coarse Latin Hypercube sampling (20 evaluations)
        # 3 levels per dimension would be 3^5=243 which is too many, so use 20 well-spread points
        n_lhs = min(20, budget)
        lhs_points = []
        for i in range(n_lhs):
            # Latin hypercube: for each dimension, divide [−5,5] into n_lhs strata
            x = np.array([(i + rng.random()) / n_lhs * 10 - 5 for _ in range(n)])
            lhs_points.append(x)

        best_x = None
        best_f = np.inf
        eval_count = 0

        for x in lhs_points:
            if not time_ok():
                break
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Stage 2: CMA-ES initialized at best point
        # Use a moderate initial sigma for the rotated Griewank-Rosenbrock
        sigma = 1.0
        x = best_x.copy()
        fx = best_f
        C = np.eye(n)  # covariance matrix
        alpha_mu = 1.0
        mu = max(1, n // 2)
        # We'll do a simplified (1+1) CMA-ES variant for efficiency

        # Actually, let's do a proper small-population CMA-ES
        pop_size = max(4, 2 * n + 1)
        # Remaining budget for CMA-ES
        remaining = budget - eval_count
        if remaining <= 0:
            return best_x, best_f

        # CMA-ES parameters
        sigma = 1.5
        C = np.eye(n)
        pc = np.zeros(n)
        cs = 1 / (n + 3)
        c1 = 2 / ((n + 1.3) ** 2 + 10)
        cmu = min(1 - c1, 2 * (mu - 1 + 1 / mu) / ((n + 2) ** 2 + 10))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        damps = 1 + 2 * max(0, np.sqrt((mu - 1 + 1 / mu) / (n + 2) - 1)) + cs
        dampc = 1 + 2 * max(0, np.sqrt((mu - 1 + 1 / mu) / (n + 2) - 1)) + cs

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / (weights ** 2).sum()

        mean = best_x.copy()
        evals_left = remaining

        while evals_left > 0 and time_ok():
            # Generate population
            W = np.linalg.cholesky(C)
            z = rng.standard_normal((pop_size, n))
            pops = mean[None, :] + sigma * (z @ W.T)
            pops = np.clip(pops, -5, 5)

            # Evaluate
            fits = []
            for px in pops:
                if not time_ok():
                    break
                pf = func(px)
                fits.append(pf)
                evals_left -= 1
                if evals_left <= 0:
                    break

            if len(fits) < pop_size:
                # Pad with inf if we ran out of budget mid-generation
                while len(fits) < pop_size:
                    fits.append(np.inf)

            # Sort by fitness
            order = np.argsort(fits)
            # Take top mu
            idx = order[:mu]
            new_mean = (weights[:, None] * pops[idx]).sum(axis=0)

            # Update evolution path
            y = (new_mean - mean) / sigma
            pc = (1 - cs) * pc + cs / np.sqrt(mu_eff) * y
            cs_norm = np.sqrt(pc @ pc) / chi_n
            if cs_norm < 1.4 + 2 / (n + 1):
                C = (1 - c1 - cmu) * C + c1 * (2 + c1) * np.outer(pc, pc) + \
                    cmu * (weights[idx] * np.outer(pops[idx] - mean, pops[idx] - mean)).sum(axis=0)

            # Update sigma (CSA)
            sigma *= np.exp(cs / damps * (cs_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            mean = new_mean

            # Track best
            if fits[idx[0]] < best_f:
                best_f = fits[idx[0]]
                best_x = pops[idx[0]].copy()

            # Check for convergence
            if sigma < 1e-8:
                break

        # Stage 3: BFGS polish
        if time_ok() and best_x is not None:
            try:
                res = minimize(func, best_x, method='BFGS', options={'maxiter': 100, 'gtol': 1e-8})
                if res.success or res.fun < best_f:
                    best_x = np.clip(res.x, -5, 5)
                    best_f = res.fun
            except Exception:
                pass

            # Try Nelder-Mead as well for robustness
            if time_ok():
                try:
                    res2 = minimize(func, best_x, method='Nelder-Mead',
                                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                    if res2.fun < best_f:
                        best_x = np.clip(res2.x, -5, 5)
                        best_f = res2.fun
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
