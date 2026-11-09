"""CMA-ES (full covariance) global search + finite-diff SQP polish for BBOB f12."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES global search + SQP local polish exploiting bent-cigar anisotropy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        deadline = t0 + 275.0  # safety margin vs 330 s kill
        max_calls = budget

        calls = [0]
        best_fx = float('inf')
        best_x = np.zeros(dim)

        def f(x):
            nonlocal best_fx, best_x
            if calls[0] >= max_calls:
                return 1e18
            calls[0] += 1
            val = float(func(x))
            if not np.isfinite(val):
                return 1e18
            if val < best_fx:
                best_fx = val
                best_x = x.copy()
            return val

        def remaining():
            return deadline - time.time()

        def finite_diff_gradient(x, h=1e-5):
            g = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                g[i] = (f(xp) - f(xm)) / (2.0 * h)
            return g

        def finite_diff_hessian(x, h=1e-4):
            H = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(i, dim):
                    xpp = x.copy(); xpm = x.copy()
                    xmp = x.copy(); xmm = x.copy()
                    xpp[i] += h; xpp[j] += h
                    xpm[i] += h; xpm[j] -= h
                    xmp[i] -= h; xmp[j] += h
                    xmm[i] -= h; xmm[j] -= h
                    H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h * h)
                    H[j, i] = H[i, j]
            return H

        def sqp_step(x, fx, g, H):
            n = dim
            reg = 1e-6
            H_reg = H + reg * np.eye(n)
            try:
                eigvals = np.linalg.eigvalsh(H_reg)
                if np.min(eigvals) < 1e-8:
                    d = -g
                    d = d / (np.linalg.norm(d) + 1e-12)
                else:
                    d = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                d = -g
                d = d / (np.linalg.norm(d) + 1e-12)

            max_step = 2.0
            step_norm = np.linalg.norm(d)
            if step_norm > max_step:
                d = d * (max_step / step_norm)

            alpha = 1.0
            c1 = 1e-4
            x_new = x
            fx_new = fx
            success = False
            for _ in range(30):
                if calls[0] >= max_calls or remaining() < 5.0:
                    break
                x_trial = np.clip(x + alpha * d, -5, 5)
                fx_trial = f(x_trial)
                if fx_trial <= fx + c1 * alpha * (g @ d):
                    x_new = x_trial
                    fx_new = fx_trial
                    success = True
                    break
                alpha *= 0.5

            if not success or alpha < 1e-10:
                d_alt = -g
                ng = np.linalg.norm(d_alt)
                if ng > 1e-12:
                    d_alt = d_alt / ng * 0.1
                    x_trial = np.clip(x + d_alt, -5, 5)
                    fx_trial = f(x_trial)
                    if fx_trial < fx:
                        x_new = x_trial
                        fx_new = fx_trial
                        success = True
            return x_new, fx_new, success

        def sqp_polish(x0, fx0, max_iter=8):
            x = x0.copy()
            fx = fx0
            for _ in range(max_iter):
                if calls[0] >= max_calls or remaining() < 10.0:
                    break
                g = finite_diff_gradient(x)
                if np.linalg.norm(g) < 1e-8:
                    break
                H = finite_diff_hessian(x)
                x_new, fx_new, ok = sqp_step(x, fx, g, H)
                if not ok:
                    break
                if fx_new >= fx - 1e-10:
                    break
                x = x_new
                fx = fx_new
            return x, fx

        # ---- CMA-ES global search with full covariance ----
        rng = np.random.default_rng(1)

        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # CMA-ES hyperparameters for dim=5
        mu = max(1, dim // 2)
        w = np.full(mu, 1.0)
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1.0 + 2 * max(0, (mueff - 2) / (dim + mueff)) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))
        lam = max(2 * mu + 1, dim + 2)

        max_cma_iter = 200
        for it in range(max_cma_iter):
            if calls[0] >= max_calls or remaining() < 30.0:
                break

            # Cholesky decomposition of C
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = 0.5 * (C + C.T)
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    B = np.eye(dim)
            B_inv = np.linalg.inv(B)

            # Sample offspring
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = mean + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate
            fvals = np.array([f(y) for y in offspring])
            order = np.argsort(fvals)
            offspring_sorted = [offspring[i] for i in order]
            fvals_sorted = fvals[order]

            # Update weighted mean
            mean_old = mean.copy()
            mean = np.zeros(dim)
            for i in range(mu):
                mean += w[i] * offspring_sorted[i]

            # Update p_sigma (evolution path for step size)
            p_sigma = (1 - c_sigma) * p_sigma
            p_sigma += np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B_inv @ (mean - mean_old))
            ps_norm = np.linalg.norm(p_sigma)
            if ps_norm > (1.4 + 1 / c_sigma) * chi_n:
                p_sigma *= (1.4 + 1 / c_sigma) * chi_n / ps_norm

            # Step size adaptation
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))
            sigma = min(sigma, 10.0)
            sigma = max(sigma, 1e-12)

            # Update p_c (evolution path for covariance)
            p_c = (1 - c_c) * p_c
            p_c += np.sqrt(c_c * (2 - c_c) * mueff) * (B_inv @ (mean - mean_old))

            # Rank-one + rank-mu covariance update
            C = (1 + c_1 * c_c * (2 - c_c) - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - c_c) * C)
            for i in range(mu):
                y_i = B_inv @ (offspring_sorted[i] - mean_old)
                C += c_mu * w[i] * np.outer(y_i, y_i)

            # Ensure symmetry and positive-definiteness
            C = 0.5 * (C + C.T)
            try:
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C = C + (1e-8 - np.min(eigvals)) * np.eye(dim)
            except np.linalg.LinAlgError:
                C = np.eye(dim)

            # Track best
            if fvals_sorted[0] < best_fx:
                best_fx = fvals_sorted[0]
                best_x = offspring_sorted[0].copy()

            # Convergence: sigma too small
            if sigma < 1e-10:
                break

        # ---- SQP polish from best point ----
        if calls[0] < max_calls and remaining() > 20.0:
            x = best_x.copy()
            fx = best_fx
            x, fx = sqp_polish(x, fx, max_iter=10)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Extra polish rounds from perturbed best if budget remains
        if calls[0] < max_calls and remaining() > 15.0:
            for _ in range(3):
                if calls[0] >= max_calls or remaining() < 10.0:
                    break
                x = best_x + 1e-3 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = f(x)
                x, fx = sqp_polish(x, fx, max_iter=5)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        if not np.all(np.isfinite(best_x)):
            best_x = np.zeros(dim)
            if calls[0] < max_calls:
                best_fx = f(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
