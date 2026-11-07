import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        X_obs = []
        y_obs = []

        def evaluate(pt):
            nonlocal calls_made, best_x, best_f, X_obs, y_obs
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            X_obs.append(pt.copy())
            y_obs.append(val)
            return val

        # Phase 1: Bayesian Optimization for first 20 evaluations
        # Initialize with 5 random points
        rng_init = np.random.default_rng(42)
        n_init = 5
        for _ in range(n_init):
            x = rng_init.uniform(-5, 5, dim)
            evaluate(x)

        # GP Parameters
        # Matern-5/2 kernel: k(x, x') = sigma_f^2 * (1 + sqrt(5)*r) * exp(-sqrt(5)*r) where r = ||x - x'|| / l
        # We need to estimate sigma_f and l. For simplicity, we can fix them or use a simple heuristic.
        # Given the function is quadratic and bounded, let's assume a length scale of ~1 and signal variance ~10.
        # A more robust approach would be to optimize hyperparameters, but that's expensive.
        # Let's use fixed hyperparameters for speed, or a very simple optimization.
        
        def matern52_kernel(X1, X2, length_scale=1.0, signal_var=10.0):
            # X1: (n1, d), X2: (n2, d)
            d = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
            r = np.sqrt(d) / length_scale
            K = signal_var * (1 + np.sqrt(5) * r) * np.exp(-np.sqrt(5) * r)
            return K

        def fit_gp(X, y, length_scale=1.0, signal_var=10.0, noise_var=1e-6):
            n = len(X)
            K = matern52_kernel(X, X, length_scale, signal_var)
            K[np.diag_indices_from(K)] += noise_var
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            return K, L, alpha

        def predict_gp(X_new, X, y, L, alpha, length_scale=1.0, signal_var=10.0):
            K_s = matern52_kernel(X_new, X, length_scale, signal_var)
            mu = K_s @ alpha
            # For EI, we only need the mean and variance
            # Var = signal_var - sum(K_s_i * L_inv * K_s_i^T)
            # We can compute the variance efficiently if we have L
            # But for simplicity in EI max search, we might just use the mean and a fixed variance or approximate.
            # Actually, for EI, we need the predictive variance.
            # Let's compute the predictive variance properly.
            # v = solve(L, K_s.T)
            # var = signal_var - sum(K_s * v, axis=1)
            v = np.linalg.solve(L, K_s.T)
            var = signal_var - np.sum(K_s * v, axis=1)
            var = np.maximum(var, 1e-10)
            return mu, var

        def expected_improvement(mu, var, f_best, xi=0.01):
            z = (f_best - mu - xi) / np.sqrt(var)
            ei = (f_best - mu - xi) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
            return ei

        n_bo = 20
        length_scale = 1.0
        signal_var = 10.0
        
        for i in range(n_init, n_bo):
            if calls_made >= budget or time.time() > deadline:
                break
            
            X = np.array(X_obs)
            y = np.array(y_obs)
            f_best = np.min(y)
            
            try:
                K, L, alpha = fit_gp(X, y, length_scale, signal_var)
            except np.linalg.LinAlgError:
                # If GP fitting fails, fall back to random sampling
                x = rng_init.uniform(-5, 5, dim)
                evaluate(x)
                continue

            # Find next point by maximizing EI
            # Use random search for EI maximization
            n_candidates = 200
            X_cand = rng_init.uniform(-5, 5, (n_candidates, dim))
            mu, var = predict_gp(X_cand, X, y, L, alpha, length_scale, signal_var)
            ei = expected_improvement(mu, var, f_best)
            idx_max = np.argmax(ei)
            x_next = X_cand[idx_max]
            evaluate(x_next)

        # Phase 2: CMA-ES
        x = best_x.copy()
        fx = best_f
        
        rng = np.random.default_rng(12345)
        sigma = 0.1
        lambda_pop = 10
        mu = 5
        weight_sum = 0.0
        for k in range(mu):
            weight_sum += 1.0 / (k + 1)
        weights = np.array([1.0 / (k + 1) for k in range(mu)])
        weights /= weight_sum
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (mu_eff - 2 + 2 / (dim + mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(c_mu, c_1)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        step_size = np.sqrt(2 * (dim + 1) * (1 - 1 / np.sqrt(2 * (dim + 1))))

        max_iter = 100
        for it in range(max_iter):
            if calls_made + lambda_pop > budget or time.time() > deadline:
                break

            # Generate population
            Z = rng.standard_normal((lambda_pop, dim))
            L_C = np.linalg.cholesky(C)
            Y = np.array([x + sigma * (Z[i] @ L_C) for i in range(lambda_pop)])
            Y = np.clip(Y, -5, 5)

            # Evaluate
            F = np.array([evaluate(Y[i]) for i in range(lambda_pop)])

            # Sort by fitness
            order = np.argsort(F)
            Z_sorted = Z[order]
            Y_sorted = Y[order]

            # Update mean
            x_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

            # Update sigma
            z_sigma = (x_new - x) / (sigma * step_size * np.sqrt(np.trace(C)))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_sigma
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) < (1 + 2 / (dim + 1)) * step_size else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / (sigma * step_size)

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1) * C + c_1 * np.outer(p_c, p_c)
            C = C + c_mu * np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu] - x, (Y_sorted[:mu] - x) / (sigma * step_size))[:, None, :] / (sigma ** 2 * step_size ** 2), axis=0)
            # Ensure symmetry
            C = (C + C.T) / 2.0

            # Update sigma
            sigma *= np.exp(c_sigma / (2 * dim) * (np.linalg.norm(np.linalg.cholesky(C)) ** 2 - dim))
            sigma = min(sigma, 1.0)

            x = x_new
            fx = F[order[0]]

            if sigma < 1e-14:
                break

        # Final polish with (1+1)-ES
        sigma = 0.01
        rng2 = np.random.default_rng(999)
        for _ in range(200):
            if calls_made >= budget or time.time() > deadline:
                break
            z = rng2.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            fy = evaluate(y)
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1
            else:
                sigma *= 0.9

        return best_x, best_f
