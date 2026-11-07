"""Bayesian Optimization with Gaussian Process surrogate for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization using a Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Latin Hypercube sampling for initial points
        n_initial = min(20, budget // 5)
        if n_initial < 5:
            n_initial = 5

        # Generate LHS points in [-5, 5]^dim
        x_init = np.zeros((n_initial, dim))
        for d in range(dim):
            idx = rng.permutation(n_initial)
            x_init[idx, d] = (idx + 0.5) / n_initial * 10.0 - 5.0

        # Evaluate initial points
        X = x_init.copy()
        Y = np.array([func(x) for x in X])
        n_calls = n_initial

        # Fixed normalization: map [-5,5] to [0,1] for stable kernel distances
        def normalize(x):
            return (x + 5.0) / 10.0

        def denormalize(x):
            return x * 10.0 - 5.0

        # Kernel function: Matern 5/2 in normalized space
        def matern_kernel(X1, X2, length_scale, variance):
            X1n = normalize(X1)
            X2n = normalize(X2)
            diff = X1n[:, np.newaxis, :] - X2n[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            sqrt5_d = np.sqrt(5.0) * dist / length_scale
            K = variance * (1.0 + sqrt5_d + sqrt5_d ** 2 / 3.0) * np.exp(-sqrt5_d)
            return K

        # Hyperparameters
        length_scale = 0.3
        variance = 1.0
        noise_var = 1e-6

        best_x = X[np.argmin(Y)].copy()
        best_y = Y.min()

        # Acquisition: Expected Improvement
        def expected_improvement(y_mean, y_std, f_best, xi=0.01):
            y_best = f_best - xi
            z = (y_best - y_mean) / (y_std + 1e-12)
            ei = (y_best - y_mean) * norm.cdf(z) + (y_std + 1e-12) * norm.pdf(z)
            ei = np.maximum(ei, 0.0)
            return ei

        # GP state: cached Cholesky and solve
        K_train = None
        L = None
        K_inv_y = None

        def rebuild_gp():
            nonlocal K_train, L, K_inv_y
            K_train = matern_kernel(X, X, length_scale, variance) + noise_var * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K_train)
            except np.linalg.LinAlgError:
                # Add jitter and retry
                jitter = 1e-8
                K_train += jitter * np.eye(len(X))
                L = np.linalg.cholesky(K_train)
            # Solve L * K_inv_y = Y using triangular solve
            K_inv_y = np.linalg.solve(L.T, np.linalg.solve(L, Y))

        # Initial GP build
        rebuild_gp()

        # GP prediction using cached Cholesky
        def gp_predict(X_test):
            X_testn = normalize(X_test)
            Xn = normalize(X)
            diff = Xn[:, np.newaxis, :] - X_testn[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            sqrt5_d = np.sqrt(5.0) * dist / length_scale
            K_test = variance * (1.0 + sqrt5_d + sqrt5_d ** 2 / 3.0) * np.exp(-sqrt5_d)
            # K_test shape: (n_train, n_test)
            # Mean: K_test.T @ K_inv_y
            mu = K_test.T @ K_inv_y
            # Variance: 1 - sum(K_test^2 * K_inv_y, axis=0)
            v = K_test * K_inv_y[np.newaxis, :]
            var = 1.0 - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-12)
            return mu, np.sqrt(var)

        # Main optimization loop
        while n_calls < budget:
            # Generate candidate points: mix of global random and local around best
            n_candidates = 150
            # 60% global random, 40% local perturbations
            n_global = int(0.6 * n_candidates)
            n_local = n_candidates - n_global

            X_cand_global = rng.uniform(-5, 5, (n_global, dim))
            # Local: perturb best_x with decreasing step size
            step_size = max(0.5, 2.0 * (1.0 - n_calls / budget))
            X_cand_local = best_x + rng.normal(0, step_size, (n_local, dim))
            X_cand_local = np.clip(X_cand_local, -5, 5)

            X_cand = np.vstack([X_cand_global, X_cand_local])

            # Predict mean and std at candidates
            mu, std = gp_predict(X_cand)

            # Compute EI
            ei = expected_improvement(mu, std, best_y)

            # Select best candidate
            idx = np.argmax(ei)
            x_new = X_cand[idx].copy()
            y_new = func(x_new)
            n_calls += 1

            # Update training set
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            # Update best
            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()

            # Rebuild GP every few iterations to keep it fresh
            # Rebuild every 1 call for accuracy, but that's expensive.
            # Instead, rebuild every 3-5 calls to balance cost and accuracy.
            if n_calls % 3 == 0 or n_calls == budget:
                rebuild_gp()

            # Adapt hyperparameters every 50 calls
            if n_calls % 50 == 0 and n_calls > n_initial:
                # Simple grid search over length_scale
                best_ls = length_scale
                best_score = np.inf
                for ls_candidate in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
                    # Evaluate GP fit quality using leave-one-out or simple R2
                    # Use a quick metric: correlation between predictions and actuals
                    Xn = normalize(X)
                    diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
                    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
                    sqrt5_d = np.sqrt(5.0) * dist / ls_candidate
                    K = 1.0 * (1.0 + sqrt5_d + sqrt5_d ** 2 / 3.0) * np.exp(-sqrt5_d)
                    K += 1e-6 * np.eye(len(X))
                    try:
                        L_try = np.linalg.cholesky(K)
                        K_inv_Y = np.linalg.solve(L_try.T, np.linalg.solve(L_try, Y))
                        # Predict training points
                        mu_pred = K @ K_inv_Y
                        # R2 score
                        ss_res = np.sum((Y - mu_pred) ** 2)
                        ss_tot = np.sum((Y - Y.mean()) ** 2) + 1e-12
                        r2 = 1.0 - ss_res / ss_tot
                        if r2 < best_score:
                            best_score = r2
                            best_ls = ls_candidate
                    except np.linalg.LinAlgError:
                        continue
                length_scale = best_ls

        return best_x, best_y

# EVOLVE-BLOCK-END
