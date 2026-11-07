# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


class Optimizer:
    """Bayesian Optimization using a Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # 1. Initialization: Latin Hypercube Sampling
        n_initial = 15
        x_init = np.zeros((n_initial, dim))
        for d in range(dim):
            idx = rng.permutation(n_initial)
            x_init[idx, d] = (idx + 0.5) / n_initial
        x_init = x_init * 10.0 - 5.0

        X = x_init.copy()
        Y = np.array([func(x) for x in X])
        n_calls = n_initial

        # 2. GP Setup
        # We will use a Matern 5/2 kernel.
        # Hyperparameters: length_scale (ls), signal_variance (sv), noise_variance (nv)
        
        # Initial guesses
        ls = 1.0
        sv = 1.0
        nv = 1e-6

        # Standardization is handled by scaling inputs to [0, 1] internally for kernel computation
        # to make hyperparameters more stable.
        # Map [-5, 5] to [0, 1]
        def scale(x):
            return (x + 5.0) / 10.0

        def unscale(x):
            return x * 10.0 - 5.0

        X_scaled = scale(X)

        def matern_kernel(X1, X2, ls, sv):
            """Compute Matern 5/2 kernel."""
            # X1, X2 are scaled to [0, 1]
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            sqrt5_d = np.sqrt(5) * dist / ls
            K = sv * (1 + sqrt5_d + sqrt5_d ** 2 / 3.0) * np.exp(-sqrt5_d)
            return K

        def log_marginal_likelihood(theta, X_train, Y_train):
            """Compute log marginal likelihood for given hyperparameters."""
            ls, sv, nv = theta
            if ls <= 0 or sv <= 0:
                return -1e10
            K = matern_kernel(X_train, X_train, ls, sv) + nv * np.eye(len(X_train))
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                log_det = 2 * np.sum(np.log(np.diag(L)))
                mll = -0.5 * Y_train @ alpha - 0.5 * log_det - 0.5 * len(Y_train) * np.log(2 * np.pi)
                return mll
            except np.linalg.LinAlgError:
                return -1e10

        def optimize_hyperparameters(X_train, Y_train, current_ls, current_sv):
            """Optimize hyperparameters using L-BFGS on log marginal likelihood."""
            # Grid search for initial ls to avoid local minima
            ls_candidates = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
            best_mll = -1e10
            best_ls = current_ls
            best_sv = current_sv
            
            # Coarse grid search
            for ls in ls_candidates:
                for sv in [0.1, 1.0, 10.0]:
                    mll = log_marginal_likelihood([ls, sv, nv], X_train, Y_train)
                    if mll > best_mll:
                        best_mll = mll
                        best_ls = ls
                        best_sv = sv
                        
            # Refine with L-BFGS
            def neg_mll(theta):
                return -log_marginal_likelihood(theta, X_train, Y_train)
            
            x0 = np.array([best_ls, best_sv, nv])
            bounds = [(0.01, 10.0), (0.01, 100.0), (1e-8, 1e-2)]
            res = minimize(neg_mll, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
            
            if res.success and res.fun < best_mll:
                best_ls = res.x[0]
                best_sv = res.x[1]
                # nv is fixed for simplicity in this refinement, or could be optimized
            return best_ls, best_sv

        def gp_predict(X_test, X_train, Y_train, ls, sv):
            """Predict mean and std at X_test."""
            K_train = matern_kernel(X_train, X_train, ls, sv) + nv * np.eye(len(X_train))
            K_test = matern_kernel(X_train, X_test, ls, sv)
            K_test_test = matern_kernel(X_test, X_test, ls, sv)
            
            try:
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                v = np.linalg.solve(L, K_test)
                
                mu = K_test.T @ alpha
                var = np.diag(K_test_test) - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-12)
                return mu, np.sqrt(var)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                K_inv_y = np.linalg.lstsq(K_train, Y_train, rcond=None)[0]
                mu = K_test.T @ K_inv_y
                var = np.diag(K_test_test) - np.sum(K_test ** 2 * K_inv_y[np.newaxis, :], axis=1)
                var = np.maximum(var, 1e-12)
                return mu, np.sqrt(var)

        def expected_improvement(X_test, mu, std, f_best, xi=0.01):
            """Compute EI."""
            y_best = f_best - xi
            z = (y_best - mu) / (std + 1e-12)
            ei = (y_best - mu) * norm.cdf(z) + (std + 1e-12) * norm.pdf(z)
            ei = np.maximum(ei, 0.0)
            return ei

        best_x = X[np.argmin(Y)]
        best_y = Y.min()

        # Main Optimization Loop
        while n_calls < budget:
            # Optimize hyperparameters every 20 steps
            if n_calls % 20 == 0:
                ls, sv = optimize_hyperparameters(X_scaled, Y, ls, sv)

            # Generate candidates for acquisition
            # Use a mix of random points and points around the best current
            n_candidates = 2000
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Add some points near the best current solution for exploitation
            n_local = 100
            X_local = best_x + rng.normal(0, 0.5, (n_local, dim))
            X_local = np.clip(X_local, -5, 5)
            X_cand = np.vstack([X_cand, X_local])
            
            X_cand_scaled = scale(X_cand)
            
            # Predict
            mu, std = gp_predict(X_cand_scaled, X_scaled, Y, ls, sv)
            
            # Acquisition
            ei = expected_improvement(X_cand, mu, std, best_y)
            
            # Select best candidate
            idx = np.argmax(ei)
            x_new = X_cand[idx]
            y_new = func(x_new)
            n_calls += 1
            
            # Update training set
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)
            X_scaled = np.vstack([X_scaled, scale(x_new)])
            
            # Update best
            if y_new < best_y:
                best_y = y_new
                best_x = x_new

        return best_x, best_y

# EVOLVE-BLOCK-END
