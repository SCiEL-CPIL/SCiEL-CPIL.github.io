"""Optimized for BBOB f11 (Discus) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0
        all_evaluated = []  # (x, f) for potential refinement

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return None
            x_arr = np.asarray(x, dtype=float)
            if x_arr.shape != (self.dim,):
                x_arr = x_arr.reshape(self.dim)
            x_clipped = np.clip(x_arr, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                return None
            if not np.isfinite(fx):
                fx = 1e10
            calls_made += 1
            all_evaluated.append((x_clipped.copy(), fx))
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def cma_es(x0, sigma0, max_iter=1000):
            """Simple CMA-ES implementation."""
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            x = x0.copy()
            sigma = sigma0

            # Population size
            lam = 4 + int(3 * np.log(dim))
            # Covariance matrix
            C = np.eye(dim)
            # Step sizes for C and D
            cs = (lambda_c + 2) / (dim + lambda_c + 5)
            ds = 1 + 2 * max(0, np.sqrt((lambda_c / (dim + 1)) - 1) - 1) + cs
            c1 = (2 / ((dim + 1.3) ** 2 + lambda_c))
            cmu = min(1 - c1, 2 * ((lambda_c / 2 - 2 + 2 / (dim + lambda_c)) / ((dim + 2) * (lambda_c + 2)) - 1))
            cc = (4 + dim / lambda_c) / (dim + 4 + 2 / lambda_c)

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            lambda_c = lam // 2

            for it in range(max_iter):
                if time.time() > deadline - 20.0 or calls_made >= self.budget:
                    break

                # Generate population
                W = np.zeros((lam, dim))
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(C))
                ys = []
                xs = []
                fs = []

                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = B @ z
                    xi = x + sigma * y
                    xi = np.clip(xi, -5.0, 5.0)
                    fi = safe_call(xi)
                    if fi is None:
                        return
                    ys.append(y)
                    xs.append(xi)
                    fs.append(fi)

                # Sort by function value
                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                xs = [xs[i] for i in order]
                fs = [fs[i] for i in order]

                # Update mean
                weights = np.ones(lam)
                weights[:lambda_c] = 2.0
                weights /= weights.sum()
                x_new = sum(w * xi for w, xi in zip(weights, xs))
                x_new = np.clip(x_new, -5.0, 5.0)

                # Update step size
                y_mean = sum(w * y for w, y in zip(weights, ys))
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * lambda_c) * y_mean / chi_n
                h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (it + 1))) < \
                           (1.4 + 2 / (dim + 1)) * chi_n else 0
                sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update covariance
                p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * lambda_c) * y_mean
                delta_h = (1 - h_sigma) * cc * (2 - cc)
                C = (1 + c1 * delta_h - c1 - cmu) * C + \
                    c1 * np.outer(p_c, p_c) + \
                    cmu * sum((w - cmu / (2 * weights[i])) * np.outer(y, y) for i, (w, y) in enumerate(zip(weights, ys)))
                # Ensure positive semi-definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                x = x_new

        def lbfgs_refine(x0, max_iter=50):
            """L-BFGS-B local refinement."""
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget or time.time() > deadline - 10.0:
                return

            def objective(x):
                return safe_call(x)

            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * self.dim,
                               options={'maxiter': max_iter, 'maxfun': 200,
                                        'ftol': 1e-12, 'gtol': 1e-8})
                if res.success and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 1: Global exploration with CMA-ES
        # Start from origin with reasonable sigma
        x0 = np.zeros(self.dim)
        fx0 = safe_call(x0)
        if fx0 is not None:
            cma_es(x0, sigma0=2.0, max_iter=200)

        # Phase 2: Try a few more starting points if budget allows
        if calls_made < self.budget - 50 and time.time() < deadline - 30.0:
            for trial in range(3):
                x_start = rng.uniform(-3, 3, self.dim)
                fx_start = safe_call(x_start)
                if fx_start is None:
                    break
                if fx_start < best_f:
                    cma_es(x_start, sigma0=1.5, max_iter=100)
                if calls_made >= self.budget - 50 or time.time() > deadline - 30.0:
                    break

        # Phase 3: L-BFGS-B refinement from best point
        if best_x is not None and calls_made < self.budget - 20 and time.time() < deadline - 10.0:
            lbfgs_refine(best_x, max_iter=30)

        # Phase 4: Final polish with small-step random search if budget remains
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 5.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(50):
                if calls_made >= self.budget or time.time() > deadline - 2.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
