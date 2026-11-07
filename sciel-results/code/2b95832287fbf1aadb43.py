"""Rewritten optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s

        # Best-so-far tracking
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Initial exploration: sample a few points to get a decent start
        rng = np.random.default_rng(42)

        # Try origin and a few random points
        candidates = [np.zeros(dim)]
        for _ in range(4):
            candidates.append(rng.uniform(-4, 4, dim))

        best_start_x = None
        best_start_f = np.inf
        for c in candidates:
            fc = func(c)
            if fc < best_start_f:
                best_start_f = fc
                best_start_x = c.copy()

        update_best(best_start_x, best_start_f)
        calls_made = len(candidates)

        # CMA-ES parameters
        lam = max(4, dim)  # population size
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (for sampling)
        # Use Cholesky decomposition
        L = np.linalg.cholesky(C)
        C_inv_sqrt = np.linalg.inv(L)  # approximate; will update

        # Step size
        sigma = 1.0
        # Learning rates
        cc = (4 + dim / dim) / (dim + 4 + 2 * dim / dim)
        cmu = min(1 - cc, 2 * (dim - 2 + 1. / dim) / ((dim + 2) ** 2 + 1))
        # Path length
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        m = best_start_x.copy()  # mean

        # Initial sigma based on spread
        sigma = 0.5

        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Adaptive restart: if no improvement for a while, restart with smaller sigma
        stall_count = 0
        max_stall = 30

        for gen in range(remaining // lam + 1):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Generate population
            w = np.zeros(lam)
            # Logarithmic weights
            w[:] = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
            w /= w.sum()
            mu = int(np.floor(np.sum(w > 0)))
            w_sum = w[:mu].sum()
            w_norm = w[:mu] / w_sum
            mu_eff = 1.0 / np.sum(w_norm ** 2)

            ys = []
            xs = []
            fs = []
            for i in range(lam):
                if time.time() > deadline or calls_made >= budget:
                    break
                z = rng.standard_normal(dim)
                # Transform: C_inv_sqrt @ z
                y = C_inv_sqrt @ z
                x_new = m + sigma * y
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                calls_made += 1
                ys.append(y)
                xs.append(x_new)
                fs.append(f_new)
                update_best(x_new, f_new)

            if len(xs) < 2:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = [ys[i] for i in idx]
            xs_sorted = [xs[i] for i in idx]
            fs_sorted = [fs[i] for i in idx]

            # Check for improvement
            if fs_sorted[0] < best_f - 1e-10:
                stall_count = 0
            else:
                stall_count += 1

            # Restart if stalled too long
            if stall_count > max_stall:
                # Reduce sigma significantly and reset some state
                sigma *= 0.5
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                C = np.eye(dim)
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L)
                m = best_x.copy()
                stall_count = 0
                max_stall = 20
                continue

            # Update mean
            m_new = np.sum([w_norm[i] * xs_sorted[i] for i in range(min(mu, len(xs_sorted)))], axis=0)

            # Update p_sigma
            y_w = np.sum([w_norm[i] * ys_sorted[i] for i in range(min(mu, len(ys_sorted)))], axis=0)
            p_sigma = (1 - cc) * p_sigma + np.sqrt(cc * (2 - cc) * mu_eff) / sigma * (m_new - m)
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cc) ** (2 * (calls_made - budget + 1) / dim)) > (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= 0.8

            # Update p_c
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cc) ** (2 * calls_made / dim)) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            p_c = (1 - cmu) * p_c + h_sigma * np.sqrt(cmu * (2 - cmu) * mu_eff) * (m_new - m) / sigma

            # Update covariance
            delta_c = (1 + h_sigma * (1 - cmu)) * cmu / sigma
            C = (1 + cmu * (2 - cmu) - cmu / dim) * C + np.outer(p_c, p_c) - delta_c * C
            # Ensure positive semi-definite
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L)

            m = m_new

            # Adaptive sigma: 1/5 rule-like
            if fs_sorted[0] < best_f:
                sigma *= 1.1
            else:
                sigma *= 0.95

            sigma = max(sigma, 1e-4)
            sigma = min(sigma, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
