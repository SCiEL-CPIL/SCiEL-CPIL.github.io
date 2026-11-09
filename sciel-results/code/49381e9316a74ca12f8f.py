"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast CMA-ES (direct Cholesky update) for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 275  # safety margin vs 300s / 330s kill

        def timed_out():
            return time.time() > deadline

        rng = np.random.default_rng(20240611)

        # ---- population / weighting (moderate for dim=5) ----
        lambda_ = 12
        mu_ = 6
        w = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # ---- CMA-ES strategy constants ----
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1.0) + c_sigma
        c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
        c_1 = 2 / (dim ** 2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))
        # Slightly increase c_1 for faster adaptation in low dimensions
        c_1 *= 1.2
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # ---- state ----
        x_mean = np.zeros(dim)
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        L = np.eye(dim)          # Cholesky: C = L L^T
        have_L = True

        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            y = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = func(y)
            call_count += 1
            if f < best_f:
                best_f = float(f)
                best_x = y.copy()
            return f

        # ---- initial diverse exploration: break the Discus ring symmetry ----
        n_init = min(20, budget)
        for k in range(n_init):
            if timed_out() or call_count >= budget:
                break
            if k == 0:
                x0 = np.zeros(dim)
            elif k == 1:
                x0 = 2.5 * rng.standard_normal(dim)
            elif k == 2:
                # Ring-like points at various radii to probe the Discus structure
                r = 3.0
                x0 = r * np.array([1.0, 0.5, -0.5, 0.3, -0.3]) / np.linalg.norm([1.0, 0.5, -0.5, 0.3, -0.3])
            elif k == 3:
                r = 1.5
                x0 = r * rng.standard_normal(dim)
                x0 = x0 / max(np.linalg.norm(x0), 1e-10) * r
            elif k == 4:
                # Axis-aligned probes
                x0 = 3.0 * np.eye(dim)[k % dim]
            else:
                # Mixed: some random, some structured
                if k % 3 == 0:
                    x0 = 3.5 * rng.uniform(-1.0, 1.0, dim)
                elif k % 3 == 1:
                    # Low-rank structured point
                    v = rng.standard_normal(dim)
                    x0 = 2.0 * v / max(np.linalg.norm(v), 1e-10) * (1.0 + 0.5 * rng.random())
                else:
                    x0 = 3.0 * rng.standard_normal(dim)
            evaluate(x0)

        # ---- main CMA-ES loop ----
        while call_count < budget:
            if timed_out():
                break

            # generate population
            offspring = []
            for _ in range(lambda_):
                if call_count >= budget or timed_out():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                f = evaluate(y)
                offspring.append((f, y, z))

            if len(offspring) < mu_:
                break

            offspring.sort(key=lambda t: t[0])
            selected = offspring[:mu_]

            # update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([w[i] * selected[i][1] for i in range(mu_)], axis=0)
            delta = x_mean - old_mean

            # evolution paths (need C^{-1/2} for p_c)
            if have_L:
                try:
                    Cinv_half_delta = np.linalg.solve(L.T, np.linalg.solve(L, delta))
                except np.linalg.LinAlgError:
                    have_L = False
                    Cinv_half_delta = delta
            else:
                Cinv_half_delta = delta

            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * Cinv_half_delta

            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * \
                np.sum([w[i] * selected[i][2] for i in range(mu_)], axis=0)

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-10, 15.0))

            # direct covariance update
            drift = 1 - np.exp(-2 * c_1 * np.dot(p_c, p_c))
            C = (1 - c_1 - c_mu) * (L @ L.T)
            C += c_1 * (drift * np.outer(p_c, p_c) + (1 - np.exp(-2 * c_sigma)) * np.eye(dim))
            for i in range(mu_):
                d = selected[i][1] - x_mean
                C += c_mu * w[i] * np.outer(d, d)
            C = 0.5 * (C + C.T)

            # refresh Cholesky, with PSD guard + drift detection
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                have_L = True
            except np.linalg.LinAlgError:
                L = L + 1e-4 * np.eye(dim)
                have_L = False

            # stagnation guard: if step size is negligible, re-seed mean
            if sigma < 1e-7:
                x_mean = best_x + 0.2 * rng.standard_normal(dim)
                sigma = 0.3

        # ---- final high-precision polish pass ----
        if call_count < budget and best_x is not None and not timed_out():
            from scipy.optimize import minimize
            remaining = budget - call_count
            if remaining > 10:
                # Use Nelder-Mead with tight tolerance for final polish
                def polish_func(x):
                    nonlocal call_count
                    if call_count >= budget or timed_out():
                        return np.inf
                    y = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                    f = func(y)
                    call_count += 1
                    if f < best_f:
                        best_f = float(f)
                        best_x = y.copy()
                    return f

                try:
                    res = minimize(
                        polish_func,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining, 200),
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                            'adaptive': True,
                        }
                    )
                    if res.fun < best_f:
                        best_f = float(res.fun)
                        best_x = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
