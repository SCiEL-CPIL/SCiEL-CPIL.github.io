"""Derivative-free optimizer using Gaussian Process surrogate + Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization with GP surrogate and EI acquisition for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _fit_gp(self, X, Y):
        n = len(X)
        d = X.shape[1]
        # Initial lengthscales: spread across the domain
        ell = np.full(d, 3.0)
        # Noise: small
        sigma_n = 1e-4

        # Build kernel matrix
        diff = X[:, None, :] - X[None, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=2))
        sqrt3_r = np.sqrt(3.0) * r
        K = np.exp(-sqrt3_r) * (1.0 + sqrt3_r)
        np.fill_diagonal(K, 0.0)  # will add sigma_n^2 to diagonal

        K += sigma_n ** 2 * np.eye(n)

        # Fit lengthscale with a few Adam steps
        log_ell = np.zeros(d)
        lr = 0.05
        for step in range(40):
            diff = X[:, None, :] - X[None, :, :]
            r = np.sqrt(np.sum(diff ** 2, axis=2))
            sqrt3_r = np.sqrt(3.0) * r
            K = np.exp(-sqrt3_r) * (1.0 + sqrt3_r)
            np.fill_diagonal(K, 0.0)
            K += sigma_n ** 2 * np.eye(n)

            # Regularize for stability
            L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            K_inv_Y = alpha

            # Marginal log-likelihood
            mll = -0.5 * Y @ K_inv_Y - 0.5 * np.sum(np.log(np.diag(L))) \
                  - 0.5 * n * np.log(2 * np.pi)

            # Gradient w.r.t. log_ell
            # dK/dlog_ell_j = -sqrt(3) * K * (d_j^2 / r) where d_j is dim j diff
            grad = np.zeros(d)
            for j in range(d):
                d_j = diff[:, :, j] ** 2
                # d_j / r, handle r=0
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = d_j / r
                ratio[~np.isfinite(ratio)] = 0.0
                dK_dlogell = -np.sqrt(3.0) * K * ratio
                # dMll/dtheta = 0.5 * alpha^T K^{-1} dK/dtheta K^{-1} alpha - 0.5 * tr(K^{-1} dK/dtheta)
                K_inv_dK = np.linalg.solve(L.T, np.linalg.solve(L, dK_dlogell))
                term1 = 0.5 * (K_inv_Y @ (K_inv_dK @ K_inv_Y))
                term2 = 0.5 * np.trace(K_inv_dK)
                grad[j] = term1 - term2

            # Adam update
            if not hasattr(self, '_adam_m'):
                self._adam_m = np.zeros(d)
                self._adam_v = np.zeros(d)
                self._adam_t = 0
            self._adam_t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            self._adam_m = beta1 * self._adam_m + (1 - beta1) * grad
            self._adam_v = beta2 * self._adam_v + (1 - beta2) * grad ** 2
            m_hat = self._adam_m / (1 - beta1 ** self._adam_t)
            v_hat = self._adam_v / (1 - beta2 ** self._adam_t)
            log_ell -= lr * m_hat / (np.sqrt(v_hat) + eps)
            log_ell = np.clip(log_ell, -2.0, 2.0)

        ell = np.exp(log_ell)

        # Final kernel
        diff = X[:, None, :] - X[None, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=2))
        sqrt3_r = np.sqrt(3.0) * r
        K = np.exp(-sqrt3_r) * (1.0 + sqrt3_r)
        np.fill_diagonal(K, 0.0)
        K += sigma_n ** 2 * np.eye(n)
        K += 1e-8 * np.eye(n)

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))

        # Normalize Y for numerical stability in predictions
        y_mean = Y.mean()
        y_std = Y.std()
        if y_std < 1e-12:
            y_std = 1.0
        Y_norm = (Y - y_mean) / y_std

        # Recompute alpha with normalized Y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))

        return {
            'X': X, 'Y': Y, 'Y_norm': Y_norm, 'y_mean': y_mean, 'y_std': y_std,
            'L': L, 'alpha': alpha, 'ell': ell, 'sigma_n': sigma_n, 'K': K
        }

    def _gp_predict(self, gp, X_new):
        """Predict mean and std at X_new using normalized GP."""
        X = gp['X']
        ell = gp['ell']
        sigma_n = gp['sigma_n']
        L = gp['L']
        alpha = gp['alpha']
        y_mean = gp['y_mean']
        y_std = gp['y_std']

        # Kernel between X_new and X
        diff = X_new[:, None, :] - X[None, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=2))
        sqrt3_r = np.sqrt(3.0) * r
        K_s = np.exp(-sqrt3_r) * (1.0 + sqrt3_r)

        # Posterior mean (in normalized space)
        v = np.linalg.solve(L, K_s.T)  # (n, n_new)
        mu_norm = (v.T @ alpha).flatten()  # (n_new,)

        # Posterior variance
        K_ss = np.exp(0) * (1.0 + 0)  # kernel at same point = 1
        var_norm = 1.0 - np.sum(v ** 2, axis=0)  # (n_new,)
        var_norm = np.maximum(var_norm, 0.0)
        std_norm = np.sqrt(var_norm)

        # Denormalize
        mu = mu_norm * y_std + y_mean
        std = std_norm * y_std

        return mu, std

    def _ei(self, mu, std, best_f, xi=0.01):
        """Expected Improvement for minimization."""
        # EI = (best_f - mu - xi) * Phi((best_f - mu - xi)/std) + std * phi((best_f - mu - xi)/std)
        z = (best_f - mu - xi) / (std + 1e-12)
        # Standard normal CDF and PDF
        Phi = 0.5 * (1.0 + np.vectorize(self._erf)(z / np.sqrt(2.0)))
        phi = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
        ei = (best_f - mu - xi) * Phi + std * phi
        return np.maximum(ei, 0.0)

    @staticmethod
    def _erf(x):
        # Simple erf approximation
        return np.vectorize(lambda t: float(np.math.erf(t)) if hasattr(np, 'math') else 0.0)(x)

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin from 300s

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        # Initialize with structured points
        init_points = [
            np.zeros(dim),
            np.ones(dim) * 3.0,
            -np.ones(dim) * 3.0,
            np.ones(dim) * 1.5,
            -np.ones(dim) * 1.5,
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-2, 2, dim),
        ]

        # Evaluate initial points
        X_data = []
        Y_data = []
        for p in init_points:
            if calls >= budget or not time_ok():
                break
            p = np.clip(p, -5, 5)
            fx = evaluate(p)
            X_data.append(p)
            Y_data.append(fx)

        if calls >= budget or not time_ok():
            return best_x, best_f

        # Fit initial GP
        X_data = np.array(X_data)
        Y_data = np.array(Y_data)
        gp = self._fit_gp(X_data, Y_data)
        n_since_fit = 0
        fit_interval = 12

        # Grid for EI maximization
        grid_size = 12  # 12 points per dimension -> 12^5 = 248832 points
        grid_pts = np.array(np.meshgrid(*[np.linspace(-5, 5, grid_size) for _ in range(dim)]))
        grid_pts = grid_pts.reshape(dim, -1).T  # (N, dim)

        # Main loop
        while calls < budget and time_ok():
            # Predict on grid
            mu, std = self._gp_predict(gp, grid_pts)

            # Compute EI
            ei = self._ei(mu, std, best_f)

            # Select best EI point
            idx = np.argmax(ei)
            x_new = grid_pts[idx].copy()

            # Add some noise to avoid exact duplicates
            x_new = x_new + 0.01 * rng.standard_normal(dim)
            x_new = np.clip(x_new, -5, 5)

            # Evaluate
            fx = evaluate(x_new)
            X_data = np.vstack([X_data, x_new.reshape(1, -1)])
            Y_data = np.append(Y_data, fx)
            n_since_fit += 1

            # Refit GP periodically
            if n_since_fit >= fit_interval and len(X_data) >= 10:
                if time_ok():
                    gp = self._fit_gp(X_data, Y_data)
                    n_since_fit = 0

        # Final local refinement around best point using (1+1)-ES
        if best_x is not None and calls < budget and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            while calls < budget and time_ok():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 0.001, 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
