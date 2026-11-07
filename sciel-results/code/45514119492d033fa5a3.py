import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    with a final high-precision L-BFGS-B polish.
    CMA-ES adapts the covariance matrix to the local landscape structure,
    which is particularly effective for rotated/shifted landscapes like BBOB functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, use 300s to be safe
        deadline = start_time + 290
        
        n = self.dim
        total_budget = self.budget
        
        # CMA-ES Parameters
        sigma = 2.0
        
        # Population size
        # For n=5, standard formula gives 8. We use 12 for better exploration.
        lambda_ = 12 
        
        # Initial mean
        mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Step sizes for parameters
        cmu = 1.0 / (4 + np.exp(n / 20))
        ccs = (4 + n / lambda_) / (n + 4 + 2 * n / lambda_)
        dsigma = 1 + 2 * max(0, np.sqrt((lambda_ / 2 - 1) / (n + 1)) - 1) + ccs
        csigma = (1 + ccs) / (n + dsigma)
        cc = (2 + 2 * np.sqrt((lambda_ / 2 - 1) / (n + 1))) / (n + 2)
        
        # State variables
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        
        # Path variables for cumulative step size adaptation
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Helper to clip to box
        def clip(x):
            return np.clip(x, -5, 5)
        
        # Evaluate function wrapper
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= total_budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_clipped = clip(x)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx
        
        # Initial evaluation of mean
        f_mean = evaluate(mean)
        
        # Main CMA-ES loop
        while calls < total_budget and time.time() < deadline:
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
            
            # Sample lambda_ points
            Z = np.random.randn(lambda_, n)
            X = mean + sigma * (Z @ L.T)
            
            # Clip population to box
            X = np.clip(X, -5, 5)
            
            # Evaluate population
            F = np.array([evaluate(x) for x in X])
            
            if calls >= total_budget:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Top mu individuals
            mu = max(1, int(lambda_ / 2))
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            
            mu_eff = 1.0 / np.sum(w**2)
            
            if np.isinf(F_sorted[0]):
                break
            
            X_top = X_sorted[:mu]
            
            # Update mean
            mean_new = np.sum(w[:, np.newaxis] * X_top, axis=0)
            
            # Update path p_sigma
            v = (mean_new - mean) / sigma
            try:
                y = np.linalg.solve(L, v)
            except np.linalg.LinAlgError:
                y = v
            p_sigma = (1 - csigma) * p_sigma + np.sqrt(csigma * (2 - csigma) * mu_eff) * y
            
            # Update step size sigma
            if np.linalg.norm(p_sigma) > 0:
                norm_p_sigma = np.linalg.norm(p_sigma)
                # Avoid division by zero
                norm_L = np.sqrt(np.trace(L @ L.T))
                if norm_L > 0:
                    exp_term = norm_p_sigma / norm_L
                    # Correction for bias
                    c_hat = np.sqrt(mu_eff) * (1 - csigma)
                    sigma *= np.exp(csigma / dsigma * (exp_term - 1 / np.sqrt(mu_eff)))
            
            # Update path p_c
            try:
                y2 = np.linalg.solve(L, (mean_new - mean) / sigma)
            except np.linalg.LinAlgError:
                y2 = (mean_new - mean) / sigma
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * y2
            
            # Update covariance matrix C
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-csigma)^{2*iter}) < (1.4 + 2/(n+1)) * E||N(0,I)||
            # Simplified check
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Expected norm of standard normal in n dimensions
            E_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(32*n**2))
            # We don't track iter count for the exact h_sigma condition easily here without a counter, 
            # but for n=5 and small budget, we can use a simplified threshold or just set h_sigma=1 
            # if the path is not too large. A robust simple heuristic:
            h_sigma = 1.0 if norm_p_sigma < 1.5 * E_norm else 0.0
            
            # C_new = (1 - cmu) * C + cmu * (sum(w_i * (x_i - m_new)(x_i - m_new)^T) + h_sigma * (p_c p_c^T - C))
            diff = X_top - mean_new
            outer = (diff.T * w) @ diff
            C = (1 - cmu) * C + cmu * (outer + h_sigma * (np.outer(p_c, p_c) - C))
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Add small regularization if needed
            try:
                _ = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(n)
            
            mean = mean_new
        
        # Final high-precision polish with L-BFGS-B
        remaining = total_budget - calls
        if remaining > 10 and np.isfinite(best_f):
            def final_func(x):
                if calls >= total_budget or time.time() > deadline:
                    return best_f + 1e6
                return evaluate(x)
            
            bounds = [(-5, 5)] * n
            try:
                from scipy.optimize import minimize
                res = minimize(final_func, best_x.copy(), method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': max(1, min(remaining // 2, 50)),
                                        'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                pass
        
        return best_x, best_f
