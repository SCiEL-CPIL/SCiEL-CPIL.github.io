import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300
        
        # 1. Initial Sampling
        # Use a mix of random and centered points to ensure coverage
        n_init = 30
        X = np.random.uniform(-5, 5, (n_init, self.dim))
        Y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_y = Y[best_idx]
        
        calls_used = n_init
        
        # 2. GP Setup
        # Estimate signal variance
        sigma_f = np.std(Y)
        if sigma_f < 1e-6:
            sigma_f = 1.0
        length_scale = 2.0
        noise_var = 1e-10
        
        def kernel(X1, X2):
            # RBF Kernel
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return sigma_f**2 * np.exp(-sq_dists / (2 * length_scale**2))
        
        def fit_gp(X_train, Y_train):
            K = kernel(X_train, X_train)
            K += noise_var * np.eye(len(X_train))
            try:
                L = np.linalg.cholesky(K)
            except:
                K += 1e-6 * np.eye(len(X_train))
                L = np.linalg.cholesky(K)
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            return L, alpha
        
        def predict_gp(x_new, L, alpha, X_train):
            # x_new is (dim,)
            k_x = kernel(x_new.reshape(1, -1), X_train).flatten()
            mu = k_x @ alpha
            v = np.linalg.solve(L, k_x)
            var = sigma_f**2 - np.dot(v, v)
            var = max(var, 1e-12)
            return mu, np.sqrt(var)
        
        def expected_improvement(x, mu_best, L, alpha, X_train):
            # x is (dim,)
            mu, sigma = predict_gp(x, L, alpha, X_train)
            if sigma < 1e-8:
                return 0.0
            z = (mu_best - mu) / sigma
            # EI = (mu_best - mu) * Phi(z) + sigma * phi(z)
            ei = (mu_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei
        
        # 3. BO Loop
        while calls_used < self.budget:
            if time.time() > deadline - 5:
                break
                
            L, alpha = fit_gp(X, Y)
            mu_best = best_y
            
            # Optimize EI with multiple restarts
            best_ei = np.inf
            best_x_next = None
            
            # Start points: current best + random
            start_points = [best_x.copy()]
            for _ in range(5):
                start_points.append(np.random.uniform(-5, 5, self.dim))
            
            for sp in start_points:
                res = minimize(
                    expected_improvement, 
                    sp, 
                    args=(mu_best, L, alpha, X), 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 30}
                )
                if res.fun < best_ei:
                    best_ei = res.fun
                    best_x_next = res.x.copy()
            
            if best_x_next is None:
                break
                
            best_x_next = np.clip(best_x_next, -5, 5)
            
            # Check for duplicates (within tolerance)
            # If too close to existing points, skip? No, just evaluate.
            
            f_new = func(best_x_next)
            X = np.vstack([X, best_x_next])
            Y = np.append(Y, f_new)
            calls_used += 1
            
            if f_new < best_y:
                best_y = f_new
                best_x = best_x_next.copy()
                
        return best_x, best_y
