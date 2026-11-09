import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Optimized CMA-ES implementation
        rng = np.random.default_rng(42)
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters tuned for dim=5, budget=500
        # Larger population for better exploration in short budget
        lam = 12
        mu = lam // 2
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Variance reduction constant
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial step size
        # Box is [-5, 5], so standard deviation of ~2 covers most of the range
        sigma = 2.0
        
        # Initial mean
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        eval_count = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        # Pre-compute constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Helper to get Cholesky factor with regularization
        def get_cholesky(C_mat):
            try:
                return np.linalg.cholesky(C_mat)
            except np.linalg.LinAlgError:
                # Regularize
                C_mat = C_mat + 1e-12 * np.eye(dim)
                return np.linalg.cholesky(C_mat)

        # Main loop
        while eval_count < budget:
            # Determine how many individuals to sample in this generation
            n_samples = min(lam, budget - eval_count)
            
            # Get Cholesky factor once per generation
            L = get_cholesky(C)
            
            ys = []
            xs = []
            fs = []
            
            for i in range(n_samples):
                z = rng.standard_normal(dim)
                y = L @ z
                x = mean + sigma * y
                f = func(x)
                eval_count += 1
                ys.append(y)
                xs.append(x)
                fs.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(ys) == 0:
                break
                
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            xs = xs[order]
            fs = fs[order]
            
            # We only update if we have at least mu points
            if len(ys) < mu:
                # If we ran out of budget mid-generation, we can't do a proper update
                # But we have already recorded the best point
                break
                
            # Use only the top mu points for recombination
            ys_sel = ys[:mu]
            xs_sel = xs[:mu]
            
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * xs_sel, axis=0)
            
            # Update evolution paths
            old_mean = mean
            mean = new_mean
            
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (new_mean - old_mean) / sigma
            
            # p_c update
            # Approximate h_sigma threshold
            threshold = (1.4 + 2.0 / (dim + 1)) * chi_n
            # Use a simplified check for h_sigma to avoid tracking exact iteration count power
            # This is a common approximation for short runs
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < threshold else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (new_mean - old_mean) / sigma
            
            # Update covariance matrix
            # y_w is the weighted mean of the selected y's
            y_w = np.sum(weights[:, np.newaxis] * ys_sel, axis=0)
            
            # Rank-1 update part
            rank1 = np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C
            
            # Rank-mu update part
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = ys_sel[i] - y_w
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1.0))
            
            # Clip sigma
            sigma = np.clip(sigma, 1e-12, 10.0)
            
            # Early stopping
            if sigma < 1e-12:
                break

        return best_x, best_f
