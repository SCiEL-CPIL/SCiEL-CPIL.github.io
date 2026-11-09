"""Bayesian Optimization with Gaussian Process surrogate for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization using GP surrogate and EI acquisition for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def check_time():
            return time.time() > deadline

        # Phase 1: Initial random sampling to build surrogate
        n_initial = min(20, budget // 5)
        X_samples = []
        Y_samples = []

        for i in range(n_initial):
            if call_count >= budget or check_time():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            call_count += 1
            X_samples.append(x.copy())
            Y_samples.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        X = np.array(X_samples)
        Y = np.array(Y_samples)

        # GP hyperparameters
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-6

        def rbf_kernel(X1, X2, ls, sv):
            """RBF kernel between two sets of points."""
            d2 = np.sum(X1 ** 2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2 ** 2, axis=1)[None, :]
            d2 = np.maximum(d2, 0)
            return sv ** 2 * np.exp(-0.5 * d2 / (ls ** 2))

        def gp_predict(X, Y, X_new, ls, sv, nv):
            """GP posterior mean and variance."""
            K = rbf_kernel(X, X, ls, sv) + nv * np.eye(len(X))
            K_new = rbf_kernel(X, X_new, ls, sv)
            try:
                K_inv_y = np.linalg.solve(K, Y)
            except np.linalg.LinAlgError:
                K_inv_y = np.linalg.lstsq(K, Y, rcond=None)[0]
            mean = K_new.T @ K_inv_y
            v = np.linalg.solve(K, K_new)
            var = sv ** 2 - np.sum(K_new * v, axis=0)
            var = np.maximum(var, 1e-10)
            return mean, var

        # Phase 2: Bayesian Optimization loop
        n_bo = budget - call_count
        for it in range(n_bo):
            if call_count >= budget or check_time():
                break

            # Generate candidate points
            n_candidates = 50
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))

            # Evaluate acquisition function (Expected Improvement)
            mu, var = gp_predict(X, Y, X_cand, length_scale, signal_var, noise_var)
            sigma = np.sqrt(var)

            # EI for minimization
            Z = (best_f - mu) / (sigma + 1e-10)
            ei = (best_f - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei = np.where(sigma > 1e-8, ei, 0)

            # Select point with maximum EI
            idx_best = np.argmax(ei)
            x_new = X_cand[idx_best]
            f_new = func(x_new)
            call_count += 1

            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            # Update training set
            X = np.vstack([X, x_new.reshape(1, -1)])
            Y = np.append(Y, f_new)

            # Adaptive hyperparameter tuning every 10 iterations
            if (it + 1) % 10 == 0 and len(X) > 5:
                # Simple adaptive: adjust length_scale based on spread of good points
                if len(Y) > 5:
                    sorted_idx = np.argsort(Y)
                    top_k = max(2, len(Y) // 5)
                    top_points = X[sorted_idx[:top_k]]
                    if top_k > 1:
                        distances = []
                        for i in range(top_k):
                            for j in range(i + 1, top_k):
                                distances.append(np.linalg.norm(top_points[i] - top_points[j]))
                        if distances:
                            new_ls = max(0.5, min(3.0, np.mean(distances) * 0.8))
                            length_scale = new_ls

                # Adjust noise based on convergence
                if len(Y) > 10:
                    recent = Y[-5:]
                    if np.std(recent) < 0.01:
                        noise_var = max(1e-8, noise_var * 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
