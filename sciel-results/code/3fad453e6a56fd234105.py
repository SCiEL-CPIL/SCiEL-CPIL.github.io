import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize
import warnings

# Suppress warnings from scipy
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 290.0  # Safety margin

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # 1. Initial Sampling
        n_init = min(10, budget)
        X = np.random.uniform(-5, 5, (n_init, dim))
        # Ensure we don't exceed budget
        n_init = min(n_init, budget)
        
        X_list = []
        y_list = []
        
        for i in range(n_init):
            x = X[i]
            fx = func(x)
            X_list.append(x)
            y_list.append(fx)
            
            if time.time() > self.deadline:
                break
            
        if not X_list:
            return np.zeros(dim), float('inf')
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        best_x = X[np.argmin(y)]
        best_f = np.min(y)
        
        calls_made = n_init
        
        # 2. Bayesian Optimization Loop
        # We use a simple GP implementation with ARD kernel
        # To avoid heavy dependencies or complex GP libraries, we implement a lightweight GP
        # or use a simplified surrogate. Given the constraint of std lib + numpy/scipy, 
        # implementing a robust GP from scratch is risky. 
        # However, SciPy doesn't have a built-in GP regressor. 
        # Let's implement a minimal GP using numpy.
        
        def kernel(X1, X2, lengthscales, sigma_f):
            # ARD Matern 5/2 or RBF. RBF is often faster to fit.
            # Let's use RBF for simplicity and speed, with ARD.
            # dist^2 = sum((x_i - x_j)^2 / l_k^2)
            # We need to be careful with dimensions.
            X1 = X1 / lengthscales
            X2 = X2 / lengthscales
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * np.dot(X1, X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            return sigma_f**2 * np.exp(-0.5 * sq_dists)

        def gp_fit(X, y, lengthscales, sigma_f, noise):
            K = kernel(X, X, lengthscales, sigma_f) + noise * np.eye(len(X))
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(np.linalg.cholesky(K).T, np.linalg.solve(L, y))
                return alpha, L
            except:
                # Fallback if singular
                K += 1e-6 * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(np.linalg.cholesky(K).T, np.linalg.solve(L, y))
                    return alpha, L
                except:
                    return None, None

        def gp_predict(X_new, X, y, alpha, L, lengthscales, sigma_f, noise):
            K_s = kernel(X_new, X, lengthscales, sigma_f)
            mu = K_s @ alpha
            # Variance calculation (simplified for acquisition, we only need mu and var)
            # v = L^{-1} K_s^T
            v = np.linalg.solve(L.T, K_s.T).T
            var = sigma_f**2 - np.sum(v**2, axis=1)
            var = np.maximum(var, 1e-8)
            return mu, np.sqrt(var)

        def acquisition(x, X, y, alpha, L, lengthscales, sigma_f, noise, best_f, xi=0.0, kappa=2.0):
            # Use UCB: mu + kappa * sigma
            # Or EI. UCB is often more stable for high-dimensional.
            mu, sigma = gp_predict(x.reshape(1, -1), X, y, alpha, L, lengthscales, sigma_f, noise)
            ucb = mu[0] + kappa * sigma[0]
            return -ucb # Minimize negative UCB

        # Hyperparameters
        # Initial lengthscales: assume isotropic, let it adapt? 
        # For Bent Cigar, anisotropy is key. 
        # We can try to optimize lengthscales, but that's expensive.
        # Alternative: Use a fixed heuristic or simple adaptation.
        # Let's try to optimize lengthscales and sigma_f every few iterations or use a fixed reasonable guess.
        # Given the strict time limit, optimizing HPs every step is too slow.
        # We will use a fixed kernel structure but maybe adapt lengthscales roughly.
        
        # Let's start with lengthscales that reflect the potential anisotropy.
        # Bent Cigar: last dim is steep. 
        # But we don't know the rotation. 
        # So we start isotropic and let the GP data guide the acquisition.
        # Actually, without optimizing HPs, the GP might not model the anisotropy well if lengthscales are fixed isotropic.
        # Let's try a simple heuristic: if we see large gradients in one direction, reduce lengthscale? 
        # That's complex. 
        # Let's stick to a well-tuned fixed kernel or a very simple HP optimization.
        
        # Simple approach: Use a fixed RBF kernel with a lengthscale that is a compromise.
        # Or, better: Use a small number of HP optimization steps at the beginning.
        
        # Let's try to optimize lengthscales once at the start using a few points.
        # If n_init > 5, we can do a quick HP fit.
        
        # For simplicity and robustness, let's use a fixed lengthscale vector.
        # If we assume the rotation is arbitrary, we can't guess the direction.
        # So we must rely on the data. 
        # If we don't optimize HPs, the GP is just a smooth interpolator.
        # This might be better than (1+1)-ES if the initial samples capture the valley.
        
        # Let's try a different strategy: 
        # Use a population-based method that is more robust to anisotropy than (1+1)-ES.
        # CMA-ES is the gold standard, but we can't use it directly if not installed? 
        # SciPy has minimize, but not CMA-ES.
        # We can implement a simplified CMA-ES or use a multi-start local optimizer.
        
        # Let's reconsider the BO approach. 
        # If we don't optimize HPs, the GP is limited.
        # Let's try a "Gaussian Process with fixed isotropic lengthscale" and see.
        # Lengthscale = 1.0 is reasonable for [-5,5].
        
        lengthscales = np.ones(dim)
        sigma_f = 1.0
        noise = 1e-6
        
        # Fit initial GP
        alpha, L = gp_fit(X, y, lengthscales, sigma_f, noise)
        if alpha is None:
             # Fallback to random search if GP fails
             for _ in range(budget - calls_made):
                 if time.time() > self.deadline: break
                 x = np.random.uniform(-5, 5, dim)
                 fx = func(x)
                 calls_made += 1
                 if fx < best_f:
                     best_f = fx
                     best_x = x
             return best_x, best_f

        kappa = 2.0 # Exploration-exploitation balance
        
        for i in range(budget - calls_made):
            if time.time() > self.deadline:
                break
                
            # Optimize acquisition function
            # Use multiple random starts for the local optimizer on the acquisition function
            best_acq_val = float('inf')
            best_acq_x = None
            
            n_starts = 5
            for _ in range(n_starts):
                x0 = np.random.uniform(-5, 5, dim)
                res = minimize(acquisition, x0, args=(X, y, alpha, L, lengthscales, sigma_f, noise, best_f, 0.0, kappa), 
                               method='L-BFGS-B', bounds=[(-5, 5)] * dim, options={'maxiter': 50})
                if res.fun < best_acq_val:
                    best_acq_val = res.fun
                    best_acq_x = res.x
            
            if best_acq_x is None:
                best_acq_x = np.random.uniform(-5, 5, dim)
                
            x_new = np.clip(best_acq_x, -5, 5)
            f_new = func(x_new)
            calls_made += 1
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new
                
            # Update GP
            X = np.vstack([X, x_new])
            y = np.append(y, f_new)
            
            # Refit GP
            alpha, L = gp_fit(X, y, lengthscales, sigma_f, noise)
            if alpha is None:
                 # If GP breaks, switch to random search
                 for _ in range(budget - calls_made):
                     if time.time() > self.deadline: break
                     x = np.random.uniform(-5, 5, dim)
                     fx = func(x)
                     calls_made += 1
                     if fx < best_f:
                         best_f = fx
                         best_x = x
                 break

        return best_x, best_f

