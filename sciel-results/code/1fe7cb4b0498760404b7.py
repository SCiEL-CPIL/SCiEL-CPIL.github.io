import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280  # 20s safety margin
        
        rng = np.random.default_rng(42)
        
        # Initial sampling
        n_init = min(10, self.budget)
        X = rng.uniform(-5, 5, (n_init, self.dim))
        Y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_f = Y[best_idx]
        
        X_list = [X]
        Y_list = [Y]
        
        # GP Hyperparameters
        # Fixed for speed, tuned for 5D box [-5,5]
        sigma_f = np.std(Y) if np.std(Y) > 1e-6 else 1.0
        # Length scale: Box size is 10. In 5D, typical distance is ~5-10.
        # l^2 in RBF is 1/(2*l^2) if using exp(-d^2/2l^2).
        # Let's use exp(-gamma * d^2). gamma ~ 1/100?
        gamma = 0.5 
        sigma_n = 1e-8
        
        def gp_predict(X_new, X_train, Y_train, gamma, sigma_f, sigma_n):
            """
            Returns mean and std for X_new given training data.
            """
            X_train = np.atleast_2d(X_train)
            Y_train = np.atleast_1d(Y_train)
            X_new = np.atleast_2d(X_new)
            
            K = self._kernel(X_train, X_train, gamma, sigma_f, sigma_n)
            K_new = self._kernel(X_train, X_new, gamma, sigma_f, 0) # No noise for prediction cov
            
            try:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X_train)))
            except np.linalg.LinAlgError:
                # Fallback to pinv if cholesky fails
                K_inv = np.linalg.pinv(K)
                mean = K_new.T @ K_inv @ Y_train
                # Variance calculation with pinv is tricky, use diagonal approx
                v = np.ones(len(X_train))
                var = sigma_f**2 - np.sum((K_new.T @ K_inv) * K_new, axis=0)
                var = np.maximum(var, 1e-8)
                return mean, np.sqrt(var)
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            mean = K_new.T @ alpha
            
            v = np.linalg.solve(L, K_new)
            var = sigma_f**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-8)
            
            return mean, np.sqrt(var)

        def _kernel(X1, X2, gamma, sigma_f, sigma_n):
            """
            RBF Kernel: sigma_f^2 * exp(-gamma * ||x1-x2||^2) + sigma_n * I
            """
            X1 = np.atleast_2d(X1)
            X2 = np.atleast_2d(X2)
            # Compute squared distances
            # X1 shape (n, d), X2 shape (m, d)
            # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y
            X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
            X2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
            dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
            dist_sq = np.maximum(dist_sq, 0) # Numerical stability
            
            K = sigma_f**2 * np.exp(-gamma * dist_sq)
            if sigma_n > 0:
                K[np.arange(len(X1)), np.arange(len(X2))] += sigma_n
            return K

        def expected_improvement(X_cand, best_f_val, mean, std):
            """
            Calculate EI for minimization.
            """
            X_cand = np.atleast_2d(X_cand)
            z = (best_f_val - mean) / std
            # Handle std ~ 0
            z = np.where(std < 1e-8, 0, z)
            ei = (best_f_val - mean) * norm.cdf(z) + std * norm.pdf(z)
            return -np.mean(ei) # Minimize negative EI

        calls_made = n_init
        current_budget = self.budget
        
        # Adaptive gamma?
        # If we find a good point, we might want to explore locally.
        # For now, fixed gamma.
        
        while calls_made < current_budget:
            if time.time() > deadline:
                break
            
            # Update GP hyperparameters occasionally?
            # For speed, we keep them fixed, but maybe adjust sigma_f based on current best
            # sigma_f = max(1.0, np.std(Y_list[-1]))
            
            # Optimize EI
            # Start from best_x or random
            best_x_opt = best_x.copy()
            
            # Multi-start for EI optimization to avoid local minima in the acquisition function
            best_ei_val = np.inf
            best_x_next = None
            
            n_restarts = 5
            for _ in range(n_restarts):
                # Start from current best or random
                if _ == 0:
                    x0 = best_x
                else:
                    x0 = rng.uniform(-5, 5, self.dim)
                
                # Define objective for minimize
                def ei_obj(x):
                    x = np.clip(x, -5, 5)
                    m, s = gp_predict(x, np.vstack(X_list), np.concatenate(Y_list), gamma, sigma_f, sigma_n)
                    return expected_improvement(x, best_f, m, s)
                
                res = minimize(ei_obj, x0, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res.fun < best_ei_val:
                    best_ei_val = res.fun
                    best_x_next = res.x
                    
            if best_x_next is None:
                best_x_next = best_x
                
            best_x_next = np.clip(best_x_next, -5, 5)
            
            # Evaluate
            f_next = func(best_x_next)
            calls_made += 1
            
            # Update data
            X_list.append(best_x_next.reshape(1, -1))
            Y_list.append(np.array([f_next]))
            
            if f_next < best_f:
                best_f = f_next
                best_x = best_x_next.copy()
                
            # Adapt gamma if we are making progress?
            # If best_f improved significantly, we might want to zoom in.
            # If not, we might want to explore more.
            # Simple adaptation: if best_f < 1.0 (assuming scale), decrease gamma (larger length scale)?
            # No, smaller gamma means larger length scale (smoother).
            # Larger gamma means smaller length scale (more local).
            # If we are near optimum, we want to be more local?
            # Actually, for step functions, the "step" size is fixed.
            # Let's stick to fixed gamma for stability.
            
            # If we have used 80% of budget, switch to local search?
            # The knowledge said GP was good. Let's keep GP until the end.
            
        return best_x, best_f
