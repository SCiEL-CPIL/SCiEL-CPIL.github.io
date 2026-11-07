"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s limit
        rng = np.random.default_rng(42)

        # Phase 1: Latin Hypercube Sampling for initial exploration
        n_init = min(50, self.budget)
        X = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_init)
            X[:, d] = (perm + 0.5) / n_init
        X = X * (self.ub - self.lb) + self.lb

        Y = np.array([func(x) for x in X])
        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_y = Y[best_idx]

        # Keep track of all evaluations
        all_X = X.copy()
        all_Y = Y.copy()
        calls_made = n_init

        # Phase 2: Bayesian Optimization with GP
        # Use a subset of points for GP to keep it fast
        max_gp_points = 80
        gp_X = all_X.copy()
        gp_Y = all_Y.copy()

        # GP kernel parameters (Matern 5/2)
        length_scale = 1.0
        variance = 1.0
        noise = 1e-6

        def matern52_kernel(X1, X2, ls, var):
            """Matern 5/2 kernel."""
            D = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2))
            sqrt3D = np.sqrt(3.0) * D / ls
            K = var * (1.0 + sqrt3D + sqrt3D ** 2 / 3.0) * np.exp(-sqrt3D)
            return K

        def gp_predict(X_new, X_train, Y_train, ls, var, noise):
            """GP prediction: mean and std."""
            K = matern52_kernel(X_train, X_train, ls, var) + noise * np.eye(len(X_train))
            Ks = matern52_kernel(X_train, X_new, ls, var)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                mu = Ks.T @ alpha
                v = np.linalg.solve(L, Ks.T)
                var_pred = var - np.sum(v ** 2, axis=0)
                std = np.sqrt(np.maximum(var_pred, 0))
            except np.linalg.LinAlgError:
                # Fallback to simple interpolation
                mu = np.ones(len(X_new)) * np.mean(Y_train)
                std = np.ones(len(X_new)) * np.std(Y_train)
            return mu, std

        def expected_improvement(x, X_train, Y_train, best_y, ls, var, noise, xi=0.01):
            """EI acquisition function."""
            x = x.reshape(1, -1)
            mu, std = gp_predict(x, X_train, Y_train, ls, var, noise)
            mu = mu[0]
            std = std[0]
            if std < 1e-10:
                return 0.0
            z = (best_y - mu - xi) / std
            ei = (best_y - mu - xi) * norm.cdf(z) + std * norm.pdf(z)
            return -ei  # Negative for minimization

        # BO loop
        while calls_made < self.budget:
            if time.time() > deadline - 30:  # 30s safety margin
                break

            # Select points for GP (use best points + recent points)
            if len(gp_X) > max_gp_points:
                # Keep best half and most recent half
                n_best = max_gp_points // 2
                n_recent = max_gp_points - n_best
                sorted_idx = np.argsort(all_Y)
                best_idx_gp = sorted_idx[:n_best]
                recent_idx_gp = np.arange(len(all_X) - n_recent, len(all_X))
                keep_idx = np.unique(np.concatenate([best_idx_gp, recent_idx_gp]))
                gp_X = all_X[keep_idx]
                gp_Y = all_Y[keep_idx]
            else:
                gp_X = all_X.copy()
                gp_Y = all_Y.copy()

            # Optimize acquisition function
            best_acq_val = -np.inf
            best_new_x = None

            # Try multiple starting points for acquisition optimization
            n_starts = 5
            for i in range(n_starts):
                if i == 0:
                    x0 = best_x
                else:
                    x0 = rng.uniform(self.lb, self.ub, self.dim)

                res = minimize(
                    lambda x: expected_improvement(x, gp_X, gp_Y, best_y, length_scale, variance, noise),
                    x0,
                    method='L-BFGS-B',
                    bounds=[(self.lb, self.ub)] * self.dim,
                    options={'maxiter': 50, 'ftol': 1e-6}
                )
                if res.success:
                    acq_val = -expected_improvement(res.x, gp_X, gp_Y, best_y, length_scale, variance, noise)
                    if acq_val > best_acq_val:
                        best_acq_val = acq_val
                        best_new_x = res.x

            if best_new_x is None:
                # Fallback: random point
                best_new_x = rng.uniform(self.lb, self.ub, self.dim)

            # Evaluate
            new_y = func(best_new_x)
            all_X = np.vstack([all_X, best_new_x.reshape(1, -1)])
            all_Y = np.append(all_Y, new_y)
            calls_made += 1

            if new_y < best_y:
                best_y = new_y
                best_x = best_new_x.copy()

            # Adaptively tune length scale based on progress
            if calls_made % 20 == 0:
                # Reduce length scale as we converge
                length_scale = max(0.5, length_scale * 0.95)

        # Phase 3: Final polish with CMA-ES around best point
        remaining = self.budget - calls_made
        if remaining > 10 and time.time() < deadline - 10:
            # Simple local search
            sigma = 0.1
            for _ in range(min(remaining, 30)):
                if time.time() > deadline - 5:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(self.dim), self.lb, self.ub)
                fy = func(y)
                calls_made += 1
                if fy < best_y:
                    best_x = y.copy()
                    best_y = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8

        return best_x, best_y
# EVOLVE-BLOCK-END
