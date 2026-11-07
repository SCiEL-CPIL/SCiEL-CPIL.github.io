"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize

class Optimizer:
    """Bayesian Optimization with Gaussian Process for Büche-Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        
        # GP Hyperparameters
        l = 1.0 
        sigma_n = 1e-4
        
        # Data storage
        X = []
        Y = []
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        def get_kernel(X1, X2):
            """Compute RBF kernel matrix between X1 (N1 x n) and X2 (N2 x n)."""
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            return np.exp(-0.5 * sq_dist / (l**2))
        
        def fit_gp(X_arr, Y_arr):
            """Fit GP and return alpha, L (Cholesky of K)."""
            N = len(X_arr)
            if N == 0:
                return None, None
            K = get_kernel(X_arr, X_arr)
            K += sigma_n * np.eye(N)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(N)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return None, None
            
            v = np.linalg.solve(L, Y_arr)
            alpha = np.linalg.solve(L.T, v)
            return alpha, L

        def predict_gp(x, alpha, L, X_arr):
            """Predict mean and variance for a single point x (shape n,)."""
            k_new = get_kernel(x.reshape(1, -1), X_arr)[0]
            mu = np.dot(k_new, alpha)
            v = np.linalg.solve(L, k_new)
            var = 1.0 - np.dot(k_new, v)
            if var < 0:
                var = 0.0
            return mu, var

        # Initialize with 5 random points
        init_points = np.random.uniform(-5, 5, (5, n))
        for x in init_points:
            f_x = func(x)
            X.append(x)
            Y.append(f_x)
            total_calls += 1
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()
        
        X = np.array(X)
        Y = np.array(Y)
        
        # Main loop
        while total_calls < budget:
            if time.time() > deadline:
                break
            
            alpha, L = fit_gp(X, Y)
            if alpha is None:
                x_new = np.random.uniform(-5, 5, n)
                f_new = func(x_new)
                total_calls += 1
                X = np.vstack([X, x_new])
                Y = np.append(Y, f_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                continue
            
            mu_best = np.min(Y)
            
            def neg_ei(x):
                x = np.array(x)
                mu, var = predict_gp(x, alpha, L, X)
                sigma = np.sqrt(var)
                if sigma < 1e-8:
                    return 0.0
                z = (mu_best - mu) / sigma
                ei = (mu_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei
            
            best_x_acq = None
            best_ei_val = -np.inf
            
            start_points = [best_x]
            for _ in range(3):
                start_points.append(np.random.uniform(-5, 5, n))
            start_points.append(np.zeros(n))
            
            for sp in start_points:
                if time.time() > deadline - 5:
                    break
                try:
                    res = minimize(neg_ei, sp, method='Nelder-Mead', options={'maxiter': 30, 'xatol': 1e-2, 'fatol': 1e-4})
                    if res.fun < best_ei_val:
                        best_ei_val = res.fun
                        best_x_acq = res.x
                except Exception:
                    pass
            
            if best_x_acq is None:
                best_x_acq = np.random.uniform(-5, 5, n)
            
            best_x_acq = np.clip(best_x_acq, -5, 5)
            
            f_new = func(best_x_acq)
            total_calls += 1
            
            X = np.vstack([X, best_x_acq])
            Y = np.append(Y, f_new)
            
            if f_new < best_f:
                best_f = f_new
                best_x = best_x_acq.copy()
        
        return best_x, best_f

# EVOLVE-BLOCK-END
