"""Surrogate-assisted optimizer for BBOB-style f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import warnings

class Optimizer:
    """Gaussian Process surrogate with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(42)

    def _rbf_kernel(self, X1, X2, l=0.3, sigma_f=1.0):
        """Compute RBF kernel matrix between X1 and X2."""
        # X1 and X2 are expected to be normalized to [0, 1]
        sq_dist = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
        sq_dist = np.maximum(sq_dist, 0.0)
        return sigma_f**2 * np.exp(-0.5 * sq_dist / (l**2))

    def _fit_gp(self, X_train, y_train):
        """
        Fit a simple GP. Returns a function that predicts mean and std.
        """
        n = len(X_train)
        if n == 0:
            return lambda X: (np.zeros(len(X)), np.ones(len(X)))
        
        # Normalize inputs to [0, 1]
        X_train_norm = (X_train + 5.0) / 10.0
        
        # Hyperparameters
        l = 0.3 
        sigma_f = 1.0
        sigma_n = 1e-4  # Small noise for interpolation
        
        K = self._rbf_kernel(X_train_norm, X_train_norm, l, sigma_f)
        K[np.diag_indices(n)] += sigma_n**2
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        except np.linalg.LinAlgError:
            # Add jitter if Cholesky fails
            K[np.diag_indices(n)] += 1e-2
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except np.linalg.LinAlgError:
                # Fallback to constant prediction
                mean_y = np.mean(y_train)
                var_y = np.var(y_train) + 1.0
                def predict_fallback(X):
                    return np.full(len(X), mean_y), np.full(len(X), np.sqrt(var_y))
                return predict_fallback

        def predict(X):
            X_norm = (X + 5.0) / 10.0
            K_star = self._rbf_kernel(X_norm, X_train_norm, l, sigma_f)
            mu = K_star @ alpha
            v = np.linalg.solve(L, K_star.T)
            var = sigma_f**2 - np.sum(v**2, axis=1)
            var = np.maximum(var, 1e-8)
            return mu, np.sqrt(var)
        
        return predict

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        
        X_list = []
        y_list = []
        
        # Initial points: Latin Hypercube Sampling for better coverage
        n_init = 20
        # Create LHS
        x_lhs = np.zeros((n_init, dim))
        for d in range(dim):
            # Stratified sampling
            points = (np.arange(n_init) + rng.random(n_init)) / n_init
            rng.shuffle(points)
            # Map to [-5, 5]
            x_lhs[:, d] = points * 10.0 - 5.0
            
        for i in range(n_init):
            x = x_lhs[i]
            y = func(x)
            X_list.append(x)
            y_list.append(y)
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        best_x = X[np.argmin(y)]
        best_y = min(y)
        
        calls_made = len(X_list)
        
        # Batch size for surrogate updates
        batch_size = 5
        
        while calls_made < budget:
            remaining = budget - calls_made
            current_batch_size = min(batch_size, remaining)
            
            # Fit GP to current data
            predict_func = self._fit_gp(X, y)
            
            # Generate candidates
            # Use a large pool of random candidates to allow global exploration
            # This lets the EI acquisition function decide where to look
            n_candidates = 2000
            candidates = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Predict mean and std for candidates
            mu_cand, sigma_cand = predict_func(candidates)
            
            # Expected Improvement (EI)
            # z = (best_y - mu) / sigma
            z = (best_y - mu_cand) / (sigma_cand + 1e-10)
            ei = (best_y - mu_cand) * norm.cdf(z) + sigma_cand * norm.pdf(z)
            
            # Select top K candidates
            indices = np.argsort(-ei)[:current_batch_size]
            x_next_batch = candidates[indices]
            
            # Evaluate all selected candidates
            y_next_batch = np.array([func(x) for x in x_next_batch])
            
            # Update data
            X = np.vstack([X, x_next_batch])
            y = np.append(y, y_next_batch)
            calls_made += current_batch_size
            
            # Update best
            for i in range(current_batch_size):
                if y_next_batch[i] < best_y:
                    best_y = y_next_batch[i]
                    best_x = x_next_batch[i]
                    
        return best_x, best_y
# EVOLVE-BLOCK-END
