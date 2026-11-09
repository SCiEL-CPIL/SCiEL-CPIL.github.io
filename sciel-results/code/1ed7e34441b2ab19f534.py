"""Surrogate-assisted optimizer for BBOB-style f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import warnings

class Optimizer:
    """Gaussian Process surrogate with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.rng = np.random.default_rng(42)

    def _rbf_kernel(self, X1, X2, l=1.0, sigma_f=1.0):
        """Compute RBF kernel matrix between X1 and X2."""
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
        x_min = -5.0
        x_max = 5.0
        X_train_norm = (X_train - x_min) / (x_max - x_min)
        
        # Hyperparameters
        l = 0.3  # Increased length scale for smoother interpolation
        sigma_f = 1.0
        sigma_n = 1e-2  # Increased noise for stability
        
        K = self._rbf_kernel(X_train_norm, X_train_norm, l, sigma_f)
        K[np.diag_indices(n)] += sigma_n**2
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        except np.linalg.LinAlgError:
            K[np.diag_indices(n)] += 1e-2
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except np.linalg.LinAlgError:
                mean_y = np.mean(y_train)
                var_y = np.var(y_train) + 1.0
                def predict_fallback(X):
                    return np.full(len(X), mean_y), np.full(len(X), np.sqrt(var_y))
                return predict_fallback

        def predict(X):
            X_norm = (X - x_min) / (x_max - x_min)
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
        
        # Initial points: Use a Latin Hypercube or stratified sampling for better coverage
        # Start with center
        x0 = np.zeros(dim)
        y0 = func(x0)
        X_list.append(x0)
        y_list.append(y0)
        
        # Add stratified random initial points
        n_init = 15
        for _ in range(n_init):
            x = rng.uniform(-4, 4, dim)  # Slightly smaller range to avoid boundaries
            y = func(x)
            X_list.append(x)
            y_list.append(y)
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        best_x = X[np.argmin(y)]
        best_y = min(y)
        
        calls_made = len(X_list)
        
        # Batch size for surrogate updates
        batch_size = 5  # Smaller batch for more frequent GP updates
        
        while calls_made < budget:
            remaining = budget - calls_made
            current_batch_size = min(batch_size, remaining)
            
            # Fit GP to current data
            predict_func = self._fit_gp(X, y)
            
            # Generate candidates
            # 1. Local perturbations around best point with adaptive step size
            # 2. Global random points
            # 3. Points near other good solutions
            
            n_local = 80
            n_global = 40
            n_near_good = 40
            
            candidates = []
            
            # Adaptive step size: start larger, shrink over time
            progress = calls_made / budget
            base_step = 1.5 * (1.0 - 0.5 * progress)
            
            # Local exploration around best
            for i in range(n_local):
                step_size = base_step * (0.5 + rng.random() * 0.5)
                step = step_size * rng.standard_normal(dim)
                x_cand = np.clip(best_x + step, -5, 5)
                candidates.append(x_cand)
            
            # Global exploration
            for _ in range(n_global):
                x_cand = rng.uniform(-5, 5, dim)
                candidates.append(x_cand)
            
            # Near other good solutions
            # Get top 3 solutions
            top_k = min(3, len(y))
            top_indices = np.argsort(y)[:top_k]
            for idx in top_indices:
                for _ in range(n_near_good // top_k):
                    step_size = base_step * 0.5
                    step = step_size * rng.standard_normal(dim)
                    x_cand = np.clip(X[idx] + step, -5, 5)
                    candidates.append(x_cand)
                
            candidates = np.array(candidates)
            
            # Predict mean and std for candidates
            mu_cand, sigma_cand = predict_func(candidates)
            
            # Expected Improvement (EI)
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
