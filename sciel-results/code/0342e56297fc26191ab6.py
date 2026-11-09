"""CMA-ES with local polish for BBOB f9 (rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer for the rotated Rosenbrock function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Reserve budget for local polish
        polish_budget = 40
        cma_budget = budget - polish_budget

        # Initialize CMA-ES parameters
        # Start with a reasonably large sigma to explore the domain
        x0 = np.zeros(dim)
        sigma = 2.0

        # Population size
        lambda_ = max(4 + int(3 * np.log(dim)), 10)
        # CMA-ES step size adaptation
        alpha_sigma = (lambda_ + 2) / (lambda_ + 5)
        # Covariance matrix adaptation
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4 - mu_eff)
        c_cov = (2 + 2 * min(1.0, c_sigma - 1.0) - mu_eff / (dim + 2)) / (
            dim + 1 + 2 * min(1.0, c_sigma + 0.5)
        )

        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 + 1.0 / (4 * dim) - 1.0 / (21 * dim ** 2))

        # Evaluate initial point
        x = x0.copy()
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_made = 1

        # Track best in population for adaptive sigma
        for gen in range(cma_budget // lambda_ + 1):
            if time.time() > deadline:
                break
            if calls_made >= cma_budget:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(lambda_):
                if calls_made >= cma_budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                ys.append(y)
                fs.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(ys) < 2:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order]
            fs_sorted = [fs[i] for i in order]

            # Update mean
            new_x = np.sum(np.array(ys_sorted[:mu]) * weights[:, None], axis=0)
            new_x = np.clip(new_x, -5, 5)

            # Update step size
            step_diff = (new_x - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                C_sqrt_inv.T @ step_diff
            )
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lambda_ + 1))) > (1.4 + 1 / dim) * chi_n:
                sigma *= np.exp(c_sigma / 5)

            # Update covariance
            diff = np.array(ys_sorted[:mu]) - x
            diff = diff / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * step_diff
            outer = np.outer(p_c, p_c)
            C = (
                (1 + c_c * (2 - c_c) - c_cov) * C
                + c_cov / (1 - (1 - c_c) ** (2 * (calls_made // lambda_ + 1))) * outer
                + c_cov * mu_eff * np.mean(diff[:, :, None] * diff[:, None, :], axis=0)
            )

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update C_sqrt_inv
            eigvals_inv = 1.0 / np.sqrt(eigvals)
            C_sqrt_inv = eigvecs @ np.diag(eigvals_inv) @ eigvecs.T

            x = new_x
            fx = best_f  # We track best_f separately

        # Phase 2: Local polish using Nelder-Mead
        if polish_budget > 5 and time.time() < deadline:
            nm_calls = [0]
            nm_limit = min(polish_budget, budget - calls_made)

            def nm_func(x_in, _calls=nm_calls, _limit=nm_limit):
                if time.time() > deadline:
                    return np.inf
                if _calls[0] >= _limit:
                    return np.inf
                _calls[0] += 1
                try:
                    return func(x_in)
                except Exception:
                    return np.inf

            try:
                res = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_limit * 3,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                        'initial_simplex_size': 0.5
                    }
                )
                polish_used = nm_calls[0]
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
