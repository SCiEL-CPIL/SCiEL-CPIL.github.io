"""Bayesian Optimization with GP surrogate for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """GP-based Bayesian optimizer with Nelder-Mead polish for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        max_calls = self.budget

        # Data storage
        X_data = []
        F_data = []

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_calls:
                return None
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                calls_made += 1
                if np.isfinite(f) and f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return None

        def gp_predict(X_new):
            """Predict mean and std at X_new using GP with RBF kernel."""
            if len(X_data) == 0:
                return np.zeros(X_new.shape[0]), np.ones(X_data[0].shape[0] if len(X_data) > 0 else 1)

            X = np.array(X_data)
            F = np.array(F_data)
            n = len(X)

            # Standardize for numerical stability
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0) + 1e-10
            Xs = (X - X_mean) / X_std
            X_new_s = (X_new - X_mean) / X_std

            # RBF kernel
            def rbf_kernel(A, B, length_scale=1.0):
                # Compute squared distances
                diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
                sq_dist = np.sum(diff ** 2, axis=-1)
                return np.exp(-sq_dist / (2 * length_scale ** 2))

            # Try multiple length scales for robustness
            best_pred_mean = None
            best_pred_std = None

            for ls in [0.5, 1.0, 2.0]:
                K = rbf_kernel(Xs, Xs, ls) + 1e-6 * np.eye(n)
                K_star = rbf_kernel(Xs, X_new_s, ls)

                try:
                    # Solve K * alpha = F
                    alpha = np.linalg.solve(K, F)
                    mu = K_star.T @ alpha
                    v = np.linalg.solve(K, K_star)
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse
                    alpha = np.linalg.pinv(K) @ F
                    mu = K_star.T @ alpha
                    v = np.linalg.pinv(K) @ K_star

                var = 1.0 - np.sum(v ** 2, axis=0)
                var = np.clip(var, 0, None)
                std = np.sqrt(var)

                if best_pred_mean is None or np.std(std) > np.std(best_pred_std):
                    best_pred_mean = mu
                    best_pred_std = std

            return best_pred_mean, best_pred_std

        def acquisition_score(X_new):
            """Expected Improvement acquisition function."""
            if len(X_data) == 0:
                # Uniform exploration
                return np.random.uniform(0, 1, X_new.shape[0])

            mu, sigma = gp_predict(X_new)
            sigma = np.clip(sigma, 1e-8, None)

            f_best = min(F_data)
            Z = (f_best - mu) / sigma
            ei = (f_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei

        def nelder_mead_polish(x0, max_iter=50):
            """Local polish using Nelder-Mead."""
            nonlocal calls_made, best_x, best_f

            def objective(x):
                if calls_made >= max_calls or time.time() > deadline:
                    return np.inf
                return safe_call(x)

            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_f and np.isfinite(res.fun):
                    best_x = np.clip(res.x, -5, 5).copy()
                    best_f = res.fun
            except Exception:
                pass

        # Phase 1: Initial random sampling
        n_init = 15
        for i in range(n_init):
            if calls_made >= max_calls or time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, self.dim)
            f = safe_call(x)
            if f is not None:
                X_data.append(x)
                F_data.append(f)

        # Phase 2: Bayesian Optimization loop
        bo_iter = 0
        while calls_made < max_calls and time.time() < deadline:
            bo_iter += 1

            # Generate candidate points
            n_candidates = 2000
            X_cand = np.random.uniform(-5, 5, (n_candidates, self.dim))

            # Add some points near current best
            if len(X_data) > 0:
                for _ in range(200):
                    jitter = np.random.normal(0, 0.5, self.dim)
                    X_cand = np.vstack([X_cand, best_x + jitter])

            X_cand = np.clip(X_cand, -5, 5)

            # Score candidates
            scores = acquisition_score(X_cand)
            idx = np.argmax(scores)
            x_next = X_cand[idx]

            # Evaluate
            f_next = safe_call(x_next)
            if f_next is None:
                break
            X_data.append(x_next)
            F_data.append(f_next)

            # Periodic local polish
            if bo_iter % 10 == 0 and time.time() < deadline - 60:
                nelder_mead_polish(best_x, max_iter=30)

        # Phase 3: Final local polish with remaining budget
        if calls_made < max_calls and time.time() < deadline - 30:
            nelder_mead_polish(best_x, max_iter=100)

        # Use any remaining budget for additional local searches
        remaining = max_calls - calls_made
        if remaining > 10 and time.time() < deadline - 20:
            # Try a few more local starts
            for _ in range(min(3, remaining // 20)):
                if time.time() > deadline - 20:
                    break
                x_start = best_x + np.random.normal(0, 0.3, self.dim)
                x_start = np.clip(x_start, -5, 5)
                nelder_mead_polish(x_start, max_iter=20)

        return best_x, best_f
# EVOLVE-BLOCK-END
