"""Optimizer for BBOB f23 (Katsuura) using Bayesian Optimization with GP surrogate."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Bayesian Optimization with GP surrogate for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        dim = self.dim
        budget = self.budget
        bounds = np.array([-5.0] * dim)
        upper = np.array([5.0] * dim)

        calls_made = 0
        x_global_best = None
        fx_global_best = np.inf

        # Storage for GP training data
        X_data = []
        Y_data = []

        def update_best(x, fx):
            nonlocal x_global_best, fx_global_best
            if fx < fx_global_best:
                fx_global_best = fx
                x_global_best = x.copy()

        def check_deadline():
            return time.time() > deadline

        # Phase 1: Latin Hypercube Sampling for initial GP data
        n_lhs = 40
        rng = np.random.default_rng(12345)
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, i] = (perm + 0.5) / n_lhs * 10.0 - 5.0

        for i in range(n_lhs):
            if calls_made >= budget or check_deadline():
                break
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            X_data.append(x.copy())
            Y_data.append(fx)

        if calls_made >= budget or check_deadline():
            return x_global_best, fx_global_best

        # GP kernel and hyperparameters
        # Use RBF kernel with learned length scales
        def rbf_kernel(X1, X2, lengthscales, variance):
            """Compute RBF kernel matrix."""
            X1 = X1[:, :, np.newaxis]
            X2 = X2[:, np.newaxis, :]
            sq_dist = np.sum((X1 - X2) ** 2 / lengthscales**2, axis=2)
            return variance * np.exp(-0.5 * sq_dist)

        def log_likelihood(nll_params, X, y):
            """Negative log marginal likelihood for GP."""
            try:
                # Parameters: log lengthscales (dim), log variance, log noise
                log_ls = nll_params[:dim]
                log_var = nll_params[dim]
                log_noise = nll_params[dim + 1]
                
                lengthscales = np.exp(log_ls)
                variance = np.exp(log_var)
                noise = np.exp(log_noise)
                
                K = rbf_kernel(X, X, lengthscales, variance)
                K[np.diag_indices_from(K)] += noise
                
                n = len(y)
                try:
                    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
                except np.linalg.LinAlgError:
                    return 1e6
                
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                nll = 0.5 * y @ alpha + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
                return nll
            except:
                return 1e6

        def gp_predict(X_new, X_train, y_train, lengthscales, variance, noise):
            """GP posterior prediction."""
            K = rbf_kernel(X_train, X_train, lengthscales, variance)
            K[np.diag_indices_from(K)] += noise
            Ks = rbf_kernel(X_train, X_new, lengthscales, variance)
            
            n = len(y_train)
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
            except np.linalg.LinAlgError:
                # Fallback: return mean and high uncertainty
                mean = np.mean(y_train) * np.ones(X_new.shape[0])
                var = np.var(y_train) * np.ones(X_new.shape[0])
                return mean, var
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            v = np.linalg.solve(L, Ks.T)
            
            mean = Ks @ alpha
            var = variance - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            
            return mean, var

        def expected_improvement(X_new, X_train, y_train, lengthscales, variance, noise, xi=0.01):
            """Expected Improvement acquisition function."""
            mean, var = gp_predict(X_new, X_train, y_train, lengthscales, variance, noise)
            sigma = np.sqrt(var)
            f_min = np.min(y_train)
            
            # Avoid division by zero
            sigma_safe = np.maximum(sigma, 1e-10)
            Z = (f_min - mean - xi) / sigma_safe
            from scipy.stats import norm
            EI = (f_min - mean - xi) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
            
            # For minimization, we want to maximize EI
            return EI

        # Main BO loop
        # Fit GP hyperparameters periodically
        fit_interval = 15  # Refit every 15 new points
        n_since_fit = 0
        lengthscales = None
        variance = None
        noise = None

        # Initial hyperparameter fit
        if len(X_data) >= 10:
            X_arr = np.array(X_data)
            y_arr = np.array(Y_data)
            # Normalize y for better conditioning
            y_mean = np.mean(y_arr)
            y_std = np.std(y_arr) + 1e-10
            y_norm = (y_arr - y_mean) / y_std
            
            init_params = np.zeros(dim + 2)
            init_params[:dim] = 0.0  # lengthscale = 1
            init_params[dim] = 0.0   # variance = 1
            init_params[dim + 1] = -10.0  # small noise
            
            res = minimize(log_likelihood, init_params, args=(X_arr, y_norm), 
                          method='L-BFGS-B', options={'maxiter': 50})
            
            if res.success:
                lengthscales = np.exp(res.x[:dim])
                variance = np.exp(res.x[dim])
                noise = np.exp(res.x[dim + 1])
            else:
                lengthscales = np.ones(dim)
                variance = 1.0
                noise = 1e-3

        while calls_made < budget:
            if check_deadline():
                break
            
            # Refit GP if needed
            if lengthscales is None or n_since_fit >= fit_interval:
                if len(X_data) >= 10:
                    X_arr = np.array(X_data)
                    y_arr = np.array(Y_data)
                    y_mean = np.mean(y_arr)
                    y_std = np.std(y_arr) + 1e-10
                    y_norm = (y_arr - y_mean) / y_std
                    
                    # Initialize from current or default
                    if lengthscales is not None:
                        init_params = np.concatenate([np.log(lengthscales + 1e-10), 
                                                       [np.log(variance + 1e-10), np.log(noise + 1e-10)]])
                    else:
                        init_params = np.zeros(dim + 2)
                        init_params[dim + 1] = -10.0
                    
                    res = minimize(log_likelihood, init_params, args=(X_arr, y_norm),
                                  method='L-BFGS-B', options={'maxiter': 30})
                    
                    if res.success and res.fun < 1e5:
                        lengthscales = np.exp(res.x[:dim])
                        variance = np.exp(res.x[dim])
                        noise = np.exp(res.x[dim + 1])
                    else:
                        # Keep old or use defaults
                        if lengthscales is None:
                            lengthscales = np.ones(dim)
                            variance = 1.0
                            noise = 1e-3
                    
                    n_since_fit = 0
            
            # Generate candidate points for acquisition
            # Use a mix of random points and points near current best
            n_candidates = 200
            X_cand = np.zeros((n_candidates, dim))
            
            # Half random, half around best
            n_rand = n_candidates // 2
            X_cand[:n_rand] = rng.uniform(bounds, upper, size=(n_rand, dim))
            
            if x_global_best is not None:
                # Points around current best with decreasing radius
                for i in range(n_rand, n_candidates):
                    radius = 0.5 + 2.0 * (i - n_rand) / max(1, n_candidates - n_rand)
                    X_cand[i] = x_global_best + radius * rng.standard_normal(dim)
                    X_cand[i] = np.clip(X_cand[i], bounds, upper)
            else:
                X_cand[n_rand:] = rng.uniform(bounds, upper, size=(n_candidates - n_rand, dim))
            
            # Evaluate acquisition function
            if lengthscales is not None and len(X_data) >= 5:
                X_arr = np.array(X_data)
                y_arr = np.array(Y_data)
                EI = expected_improvement(X_cand, X_arr, y_arr, lengthscales, variance, noise)
            else:
                # Fallback: use random
                EI = rng.random(n_candidates)
            
            # Select best candidate
            idx_best = np.argmax(EI)
            x_new = X_cand[idx_best].copy()
            
            # Evaluate
            fx_new = func(x_new)
            calls_made += 1
            update_best(x_new, fx_new)
            X_data.append(x_new)
            Y_data.append(fx_new)
            n_since_fit += 1
            
            if check_deadline():
                break

        return x_global_best, fx_global_best
# EVOLVE-BLOCK-END
