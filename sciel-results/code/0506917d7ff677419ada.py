"""Optimizer using Bayesian Optimization with GP surrogate and Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization with GP surrogate and Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # GP hyperparameters
        length_scales = np.ones(dim)
        noise_var = 1e-6
        signal_var = 1.0

        # Data storage
        X = []
        Y = []

        # Global best
        x_best = None
        f_best = np.inf
        calls = 0

        def _eval(x):
            nonlocal calls, x_best, f_best
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x))
            calls += 1
            if val < f_best:
                f_best = val
                x_best = x.copy()
            return val

        def _gp_predict(X_train, Y_train, X_pred):
            """Simple GP prediction with RBF kernel."""
            n_train = X_train.shape[0]
            n_pred = X_pred.shape[0]

            # Compute kernel matrix K(X_train, X_train)
            # For efficiency with 5D, use scaled RBF
            diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            # Scale by inverse of length scales squared
            sq_dist_scaled = sq_dist / (length_scales ** 2)
            K = signal_var * np.exp(-0.5 * sq_dist_scaled) + noise_var * np.eye(n_train)

            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(n_train))

            # Solve for alpha
            alpha = np.linalg.solve(np.linalg.cholesky(K).T,
                                    np.linalg.solve(L, Y_train))

            # Predictions
            K_s = signal_var * np.exp(-0.5 * np.sum(
                (X_pred[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2,
                axis=2) / (length_scales ** 2))

            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = signal_var - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 0.0)
            std = np.sqrt(var)

            return mu, std

        def _ei_acquisition(X_candidates, mu, std, f_best_val):
            """Expected Improvement acquisition function."""
            Z = (f_best_val - mu) / std
            Z = np.clip(Z, -10, 10)  # Numerical stability
            from scipy.stats import norm
            phi_Z = norm.pdf(Z)
            Phi_Z = norm.cdf(Z)
            EI = (f_best_val - mu) * Phi_Z + std * phi_Z
            # Handle where std is very small
            EI = np.where(std < 1e-8, 0.0, EI)
            return EI

        def _acquire_next_point(X_data, Y_data, f_best_val, n_candidates=200):
            """Generate candidate points and select the one with max EI."""
            n_data = len(X_data)

            # Generate candidates: mixture of random and local around best
            candidates = []

            # Random candidates
            n_random = int(n_candidates * 0.5)
            candidates.append(rng.uniform(-5, 5, size=(n_random, dim)))

            # Local candidates around best
            if x_best is not None:
                n_local = int(n_candidates * 0.3)
                # Scale local search
                local_scale = 2.0 / (1 + 0.1 * n_data)
                local_candidates = x_best + rng.standard_normal((n_local, dim)) * local_scale
                local_candidates = np.clip(local_candidates, -5, 5)
                candidates.append(local_candidates)

            # Candidates around other good points
            if n_data >= 3:
                n_other = n_candidates - n_random - int(n_candidates * 0.3)
                if n_other > 0:
                    # Pick a few other points
                    other_indices = rng.choice(n_data, size=min(3, n_data), replace=False)
                    other_candidates = []
                    for idx in other_indices:
                        n_per = max(1, n_other // len(other_indices))
                        scale = 1.5 / (1 + 0.05 * n_data)
                        pts = X_data[idx] + rng.standard_normal((n_per, dim)) * scale
                        pts = np.clip(pts, -5, 5)
                        other_candidates.append(pts)
                    candidates.append(np.vstack(other_candidates))

            X_candidates = np.vstack(candidates)
            X_candidates = np.clip(X_candidates, -5, 5)

            # Predict
            mu, std = _gp_predict(np.array(X_data), np.array(Y_data), X_candidates)

            # Compute EI
            EI = _ei_acquisition(X_candidates, mu, std, f_best_val)

            # Select best candidate
            best_idx = np.argmax(EI)
            return X_candidates[best_idx]

        # Phase 1: Initial random sampling to build GP model
        n_init = min(15, budget)
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = _eval(x)
            X.append(x)
            Y.append(f)

        if x_best is None:
            x_best = np.array(X[0])
            f_best = Y[0]

        # Phase 2: Bayesian Optimization loop
        gp_retrain_interval = 5
        last_retrain = 0
        stagnation_count = 0
        prev_best = f_best

        while calls < budget and time.time() < deadline:
            # Generate next point using EI acquisition
            x_next = _acquire_next_point(X, Y, f_best, n_candidates=200)
            f_next = _eval(x_next)
            X.append(x_next)
            Y.append(f_next)

            # Check for improvement
            if f_best < prev_best - 1e-6:
                stagnation_count = 0
                prev_best = f_best
            else:
                stagnation_count += 1

            # If stagnating, try a random restart
            if stagnation_count >= 10 and calls < budget:
                x_restart = rng.uniform(-5, 5, dim)
                f_restart = _eval(x_restart)
                X.append(x_restart)
                Y.append(f_restart)
                stagnation_count = 0
                prev_best = f_best

            # Periodically, we could retrain/adapt GP, but for simplicity
            # we keep the same hyperparameters. The GP naturally adapts
            # as more data is added.

        return x_best, f_best
# EVOLVE-BLOCK-END
