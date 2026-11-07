import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(D)))
        # For D=5, lam = 4 + 3*1.609 = 8.8 -> 8. Let's use 10 for better exploration in 5D.
        lam = 10
        
        # Weights
        mu = lam // 2
        w = np.zeros(lam)
        w[:mu] = np.log(lam/2 + 1) - np.log(np.arange(1, mu + 1))
        w_sum = np.sum(w)
        w /= w_sum
        mueff = (w * w).sum() ** -1
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (D + mueff + 5)
        c_c = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        c_1 = 2 / ((D + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((D + 2) ** 2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1) - 1)) + c_sigma
        
        # State variables
        x_mean = np.zeros(D)
        sigma = 2.0
        C = np.eye(D)
        p_c = np.zeros(D)
        p_sigma = np.zeros(D)
        
        # Cholesky factor of C
        C_cholesky = np.linalg.cholesky(C)
        
        calls_made = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        # Pre-generate random numbers if possible? No, standard normal is fine.
        
        while calls_made < budget:
            # Generate population
            ys = np.random.multivariate_normal(np.zeros(D), np.eye(D), size=lam)
            # Note: Using standard normal and multiplying by cholesky is more efficient/robust
            # ys = np.random.randn(lam, D)
            # zs = ys @ C_cholesky.T
            
            # Actually, let's use standard normal and transform
            ys = np.random.randn(lam, D)
            zs = ys @ C_cholesky.T
            
            xs = x_mean[None, :] + sigma * zs
            # Clip to bounds
            xs = np.clip(xs, -5, 5)
            
            # Evaluate
            fs = np.array([func(x) for x in xs])
            calls_made += lam
            
            # Check for NaNs or Infs (though problem says func returns float)
            # If any are bad, replace with large value
            fs[~np.isfinite(fs)] = 1e10
            
            # Sort by fitness
            order = np.argsort(fs)
            xs = xs[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
            
            # Update mean
            x_mean_new = np.sum(w[:mu, None] * xs[:mu], axis=0)
            
            # Update evolution paths
            # Normalized difference
            diff = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_cholesky.T @ diff
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls_made + 1)) / np.sqrt(np.trace(C))) < (1.4 + 2 / (D + 1)) * np.sqrt(c_sigma + (2 - c_sigma) * mueff)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * diff
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c) * np.trace(C)
            rank_mu = (C_cholesky.T @ (w[:mu, None] * (xs[:mu] - x_mean) / sigma).T).T
            C = (1 + c_1 + c_mu - c_1 - c_c + delta_h) * C \
                - c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * delta_h * C) \
                + c_mu * np.outer(rank_mu, rank_mu)
            
            # Symmetrize C to prevent drift
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(np.trace(C)) - 1))
            
            # Update mean
            x_mean = x_mean_new
            
            # Update Cholesky factor
            try:
                C_cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes singular, regularize
                C = C + 1e-6 * np.eye(D)
                C_cholesky = np.linalg.cholesky(C)
                
            # If budget is not exhausted but we have less than lam calls left, 
            # we can't do a full generation. 
            # The problem says "More than 500 calls... makes result invalid".
            # So we must stop exactly when calls_made >= budget.
            # The loop condition handles this.
            
        return best_x, best_f
