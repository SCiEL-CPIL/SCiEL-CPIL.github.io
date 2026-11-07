import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320
        n = self.dim
        B = self.budget
        
        # Data storage
        X = np.zeros((0, n))
        Y = np.zeros(0)
        
        best_x = None
        best_f = np.inf
        calls = 0

        # 1. Initialization: Latin Hypercube Sampling
        n_init = 10
        # Simple LHS
        X_init = np.random.default_rng(0).permutation(n_init)[:, None] / n_init
        X_init = X_init * 10 - 5 # Map to [-5, 5]
        X_init = np.clip(X_init, -5, 5)
        
        for i in range(n_init):
            if calls >= B: break
            x = X_init[i]
            f = func(x)
            calls += 1
            X = np.vstack([X, x])
            Y = np.append(Y, f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if calls >= B:
            return best_x, best_f

        # GP Parameters
        # Matern 5/2 kernel is robust
        def matern52_kernel(X1, X2, l, sigma_f):
            d = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=-1))
            sqrt5d = np.sqrt(5) * d / l
            K = (1 + sqrt5d + 5/3 * sqrt5d**2) * np.exp(-sqrt5d) * sigma_f**2
            return K

        def fit_gp(X, Y):
            # Hyperparameters: length_scale, sigma_f, noise
            # We can use fixed hyperparameters for simplicity or optimize them.
            # Optimizing hyperparameters at each step is expensive.
            # Let's use fixed reasonable values for n=5, box [-5,5].
            l = 1.0
            sigma_f = 1.0
            noise = 1e-6
            
            K = matern52_kernel(X, X, l, sigma_f)
            K[np.diag_indices_from(K)] += noise
            
            # Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter if singular
                K[np.diag_indices_from(K)] += 1e-4
                L = np.linalg.cholesky(K)
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            return l, sigma_f, noise, L, alpha

        def gp_predict(X_new, l, sigma_f, noise, L, alpha):
            K_star = matern52_kernel(X_new, X, l, sigma_f)
            K_ss = matern52_kernel(X_new, X_new, l, sigma_f)
            
            v = np.linalg.solve(L, K_star.T)
            mu = K_star @ alpha
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            return mu, np.sqrt(var)

        def expected_improvement(x, f_best, mu, sigma):
            # x is a single point (n,)
            # mu, sigma are arrays for the point
            if sigma < 1e-8:
                return 0.0
            z = (f_best - mu) / sigma
            # Standard normal CDF and PDF
            from scipy.stats import norm
            EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return -EI # Negate because we want to minimize

        # Main Loop
        while calls < B:
            if time.time() - start_time > time_limit:
                break
                
            # Fit GP
            try:
                l, sigma_f, noise, L, alpha = fit_gp(X, Y)
            except Exception:
                # Fallback to random if GP fails
                x = np.random.uniform(-5, 5, n)
                f = func(x)
                calls += 1
                X = np.vstack([X, x])
                Y = np.append(Y, f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                continue

            # Define acquisition function
            def acq(x):
                x = x.reshape(1, -1)
                mu, sigma = gp_predict(x, l, sigma_f, noise, L, alpha)
                return expected_improvement(x, best_f, mu[0], sigma[0])

            # Optimize acquisition function
            # Use multiple starts to avoid local maxima of EI
            best_acq_val = np.inf
            best_acq_x = None
            
            # Starts: best current point, center, random
            starts = [best_x, np.zeros(n)]
            for _ in range(5):
                starts.append(np.random.uniform(-5, 5, n))
                
            for s in starts:
                res = minimize(acq, s, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 100})
                if res.fun < best_acq_val:
                    best_acq_val = res.fun
                    best_acq_x = res.x

            if best_acq_x is None:
                best_acq_x = np.random.uniform(-5, 5, n)
            
            # Clip to box
            best_acq_x = np.clip(best_acq_x, -5, 5)
            
            # Evaluate
            f = func(best_acq_x)
            calls += 1
            
            X = np.vstack([X, best_acq_x])
            Y = np.append(Y, f)
            
            if f < best_f:
                best_f = f
                best_x = best_acq_x.copy()

        return best_x, best_f
