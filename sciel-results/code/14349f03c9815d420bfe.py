"""Bayesian Optimization with Gaussian Process surrogate for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Bayesian Optimization using a Gaussian Process surrogate model.
    Uses Expected Improvement acquisition to balance exploration and exploitation
    for the highly multimodal Weierstrass function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0
        rng = np.random.default_rng(42)

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                val = float(func(x_clipped))
                return val, x_clipped
            except Exception:
                return None, None

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                return True
            return False

        # --- GP Surrogate Implementation ---
        def gp_predict(X, X_train, y_train, length_scale, signal_variance, noise_variance):
            """
            Predict mean and variance at points X using GP trained on X_train, y_train.
            Uses RBF kernel.
            """
            n_train = X_train.shape[0]
            n_pred = X.shape[0]

            # Compute kernel matrix K(X_train, X_train)
            K_train = np.zeros((n_train, n_train))
            for i in range(n_train):
                for j in range(n_train):
                    diff = X_train[i] - X_train[j]
                    dist_sq = np.dot(diff, diff)
                    K_train[i, j] = signal_variance * np.exp(-0.5 * dist_sq / length_scale**2)
            
            # Add noise
            K_train += noise_variance * np.eye(n_train)

            # Compute kernel matrix K(X_pred, X_train)
            K_pred_train = np.zeros((n_pred, n_train))
            for i in range(n_pred):
                for j in range(n_train):
                    diff = X[i] - X_train[j]
                    dist_sq = np.dot(diff, diff)
                    K_pred_train[i, j] = signal_variance * np.exp(-0.5 * dist_sq / length_scale**2)

            # Solve for weights
            try:
                L = np.linalg.cholesky(K_train + 1e-8 * np.eye(n_train))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K_train + 1e-8 * np.eye(n_train), y_train, rcond=None)[0]

            # Predict mean
            mean = K_pred_train @ alpha

            # Predict variance
            v = np.linalg.solve(L, K_pred_train.T)
            var = signal_variance - np.sum(v**2, axis=1)
            var = np.maximum(var, 1e-10)

            return mean, var

        def expected_improvement(mean, std, best_f, xi=0.01):
            """
            Compute Expected Improvement acquisition function.
            """
            std = np.maximum(std, 1e-10)
            improvement = best_f - mean - xi
            z = improvement / std
            # Standard normal CDF and PDF
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            return np.maximum(ei, 0)

        # Initialize with a few random points
        n_init = min(10, budget)
        X_train = np.zeros((n_init, dim))
        y_train = np.zeros(n_init)

        for i in range(n_init):
            if n_calls >= budget or check_time():
                break
            x_new = rng.uniform(-5, 5, dim)
            f_new, x_eval = call_func(x_new)
            if f_new is None:
                break
            X_train[i] = x_eval
            y_train[i] = f_new
            update_best(x_eval, f_new)

        if best_x is None:
            return None

        # GP hyperparameters (will be adapted)
        length_scale = 1.0
        signal_variance = np.var(y_train) if n_init > 1 else 1.0
        signal_variance = max(signal_variance, 0.1)
        noise_variance = 1e-6

        # Main BO loop
        n_bo_iter = 0
        max_bo_iter = budget - n_calls
        consecutive_failures = 0

        while n_calls < budget and not check_time() and n_bo_iter < max_bo_iter:
            n_bo_iter += 1

            # Generate candidate points using Latin Hypercube Sampling
            n_candidates = 500
            X_candidates = rng.uniform(-5, 5, (n_candidates, dim))

            # Predict mean and variance
            try:
                mean_pred, var_pred = gp_predict(X_candidates, X_train, y_train, length_scale, signal_variance, noise_variance)
            except Exception:
                # Fallback to random if GP fails
                x_new = rng.uniform(-5, 5, dim)
                f_new, x_eval = call_func(x_new)
                if f_new is None:
                    break
                update_best(x_eval, f_new)
                continue

            # Compute EI
            ei = expected_improvement(mean_pred, np.sqrt(var_pred), best_f, xi=0.01)

            # Select point with maximum EI
            idx = np.argmax(ei)
            x_new = X_candidates[idx]

            # Evaluate
            f_new, x_eval = call_func(x_new)
            if f_new is None:
                break

            # Update best
            improved = update_best(x_eval, f_new)

            # Add to training set
            X_train = np.vstack([X_train, x_eval])
            y_train = np.append(y_train, f_new)

            # Adapt hyperparameters based on improvement
            if improved:
                consecutive_failures = 0
                # Slightly increase length scale to explore more broadly
                length_scale *= 1.05
            else:
                consecutive_failures += 1
                # Decrease length scale to exploit more locally
                length_scale *= 0.95
                length_scale = max(length_scale, 0.1)

            # Update signal variance
            if len(y_train) > 1:
                signal_variance = np.var(y_train)
                signal_variance = max(signal_variance, 0.1)

            # If too many consecutive failures, restart with random exploration
            if consecutive_failures > 20:
                consecutive_failures = 0
                # Inject random points to escape local optima
                n_random = 5
                for _ in range(n_random):
                    if n_calls >= budget or check_time():
                        break
                    x_rand = rng.uniform(-5, 5, dim)
                    f_rand, x_rand_eval = call_func(x_rand)
                    if f_rand is not None:
                        update_best(x_rand_eval, f_rand)
                        X_train = np.vstack([X_train, x_rand_eval])
                        y_train = np.append(y_train, f_rand)
                # Reset GP
                length_scale = 1.0
                signal_variance = np.var(y_train) if len(y_train) > 1 else 1.0

        # Final local polish with (1+1)-ES if budget remains
        if n_calls < budget and not check_time():
            remaining = budget - n_calls
            if remaining > 10:
                x = best_x.copy()
                f = best_f
                sigma = 0.5
                stagnant = 0
                for _ in range(remaining):
                    if n_calls >= budget or check_time():
                        break
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy, y_eval = call_func(y)
                    if fy is None:
                        break
                    if fy < f:
                        x = y_eval
                        f = fy
                        sigma *= 1.1
                        stagnant = 0
                        update_best(x, f)
                    else:
                        sigma *= 0.8
                        stagnant += 1
                        if stagnant > 20:
                            break
                    sigma = max(sigma, 1e-10)
                    sigma = min(sigma, 5.0)

        return best_x, best_f

# EVOLVE-BLOCK-END
