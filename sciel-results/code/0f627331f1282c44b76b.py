"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial exploration: random points + center
        n_init = min(10, budget // 5)
        X = np.zeros((n_init, dim))
        X[0] = np.zeros(dim)  # center
        X[1:] = rng.uniform(-5, 5, (n_init - 1, dim))

        Y = np.array([func(x) for x in X])

        best_x = X[np.argmin(Y)]
        best_y = Y.min()

        # GP hyperparameters
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-8

        def gp_kernel(X1, X2, length_scale, signal_var):
            """Matern 5/2 kernel."""
            # Compute squared distances
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            r = np.sqrt(sq_dist)
            # Matern 5/2
            sqrt3_r = np.sqrt(3) * r / length_scale
            K = signal_var * (1 + sqrt3_r + (sqrt3_r ** 2) / 3) * np.exp(-sqrt3_r)
            return K

        def gp_predict(X_new, X_obs, Y_obs, length_scale, signal_var, noise_var):
            """Predict mean and variance at X_new given observations."""
            K = gp_kernel(X_obs, X_obs, length_scale, signal_var)
            K[np.diag_indices_from(K)] += noise_var
            K_star = gp_kernel(X_obs, X_new, length_scale, signal_var)
            K_ss = gp_kernel(X_new, X_new, length_scale, signal_var)

            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)

            mean = K_star.T @ K_inv @ Y_obs
            var = np.diag(K_ss) - np.sum(K_star.T @ K_inv * K_star, axis=0)
            var = np.maximum(var, 1e-12)
            return mean, var

        # Acquire function: Expected Improvement
        def expected_improvement(X_new, X_obs, Y_obs, length_scale, signal_var, noise_var, best_y):
            mean, var = gp_predict(X_new, X_obs, Y_obs, length_scale, signal_var, noise_var)
            std = np.sqrt(var)
            eps = 1e-8
            z = (best_y - mean) / np.maximum(std, eps)
            EI = (best_y - mean) * norm.cdf(z) + std * norm.pdf(z)
            return EI

        # Adaptive hyperparameter tuning (simple)
        # We'll do a few rounds of tuning by maximizing log-marginal-likelihood
        def log_marginal_likelihood(Y_obs, X_obs, length_scale, signal_var, noise_var):
            K = gp_kernel(X_obs, X_obs, length_scale, signal_var)
            K[np.diag_indices_from(K)] += noise_var
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return -1e10
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_obs))
            lml = -0.5 * Y_obs @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(Y_obs) * np.log(2 * np.pi)
            return lml

        # Main loop
        calls_made = n_init
        for step in range(budget - n_init):
            # Occasionally tune hyperparameters
            if step % 50 == 0 and len(Y) > 10:
                # Try a few random hyperparameter settings
                best_lml = -np.inf
                best_params = (length_scale, signal_var, noise_var)
                for _ in range(5):
                    ls = length_scale * (0.5 + rng.random())
                    sv = signal_var * (0.5 + rng.random())
                    nv = noise_var * (0.1 + rng.random() * 10)
                    lml = log_marginal_likelihood(Y, X, ls, sv, nv)
                    if lml > best_lml:
                        best_lml = lml
                        best_params = (ls, sv, nv)
                length_scale, signal_var, noise_var = best_params

            # Generate candidate points for acquisition
            n_cand = 200
            candidates = rng.uniform(-5, 5, (n_cand, dim))

            # Also add some local perturbations around best_x
            n_local = 50
            local_pert = best_x + rng.normal(0, 0.5, (n_local, dim))
            local_pert = np.clip(local_pert, -5, 5)
            candidates = np.vstack([candidates, local_pert])

            # Compute EI for all candidates
            ei = expected_improvement(candidates, X, Y, length_scale, signal_var, noise_var, best_y)

            # Select best candidate
            idx = np.argmax(ei)
            x_new = candidates[idx]
            y_new = func(x_new)
            calls_made += 1

            # Update observations
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            if y_new < best_y:
                best_y = y_new
                best_x = x_new

        return best_x, best_y
# EVOLVE-BLOCK-END
