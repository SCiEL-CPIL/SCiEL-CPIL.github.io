# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import qmc


class Optimizer:
    """Gaussian Process Bayesian Optimizer with Expected Improvement acquisition and active hyperparameter tuning."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_log_likelihood(self, X, y, length_scale, noise):
        """Compute the log marginal likelihood of the GP."""
        n = X.shape[0]
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        K = np.exp(-0.5 * sq_dist / (length_scale ** 2)) + noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10 # Invalid hyperparameters
        
        # Log determinant
        log_det = 2 * np.sum(np.log(np.diag(L)))
        
        # Quadratic form
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        quad = y @ alpha
        
        log_lik = -0.5 * quad - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
        return log_lik

    def _optimize_hyperparams(self, X, y, current_ls, current_noise):
        """Optimize GP hyperparameters by maximizing log marginal likelihood."""
        def objective(params):
            ls = np.exp(params[0]) # Ensure positive
            noise = np.exp(params[1]) # Ensure positive
            ll = self._gp_log_likelihood(X, y, ls, noise)
            return -ll # Minimize negative log-likelihood

        # Initial guess in log space
        x0 = np.array([np.log(max(current_ls, 1e-3)), np.log(max(current_noise, 1e-6))])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            new_ls = np.exp(res.x[0])
            new_noise = np.exp(res.x[1])
            
            # Clamp to reasonable bounds
            new_ls = np.clip(new_ls, 0.1, 5.0)
            new_noise = np.clip(new_noise, 1e-6, 1.0)
            
            return new_ls, new_noise
        except Exception:
            return current_ls, current_noise

    def _gp_predict(self, X_new, X, y, length_scale, noise, L_cache=None):
        """
        Compute GP mean and variance at X_new.
        If L_cache is provided, it is the Cholesky factor of K.
        """
        n = X.shape[0]
        
        if L_cache is None:
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            K = np.exp(-0.5 * sq_dist / (length_scale ** 2)) + noise * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-8 * np.eye(n)
                L = np.linalg.cholesky(K)
        else:
            L = L_cache

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        diff_new = X_new[:, np.newaxis, :] - X[np.newaxis, :, :]
        sq_dist_new = np.sum(diff_new ** 2, axis=-1)
        K_star = np.exp(-0.5 * sq_dist_new / (length_scale ** 2))
        
        mean = K_star @ alpha
        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        
        return mean, var

    def _expected_improvement(self, y_best, mean, var, xi=0.0):
        """Compute Expected Improvement acquisition function."""
        sigma = np.sqrt(var)
        z = (y_best - mean - xi) / sigma
        ei = (y_best - mean - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Initial points: Use Sobol sequence for better space filling
        n_init = min(15, budget)
        sampler = qmc.Sobol(d=dim, scramble=True)
        init_points = sampler.random(n_init)
        X = init_points * 10.0 - 5.0 # Map to [-5, 5]
        
        y = np.array([func(x) for x in X])
        
        # Initial hyperparameters
        length_scale = 1.0
        noise = 1e-5
        
        # Track best
        y_best = np.min(y)
        x_best = X[np.argmin(y)]
        
        # Cache Cholesky factor
        def compute_L():
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            K = np.exp(-0.5 * sq_dist / (length_scale ** 2)) + noise * np.eye(X.shape[0])
            try:
                return np.linalg.cholesky(K)
            except:
                K += 1e-8 * np.eye(X.shape[0])
                return np.linalg.cholesky(K)
        
        L = compute_L()

        # Bayesian optimization loop
        for i in range(n_init, budget):
            # Periodically optimize hyperparameters
            if i > n_init and (i - n_init) % 10 == 0:
                new_ls, new_noise = self._optimize_hyperparams(X, y, length_scale, noise)
                if new_ls != length_scale or new_noise != noise:
                    length_scale = new_ls
                    noise = new_noise
                    L = compute_L() # Recompute Cholesky with new hyperparams

            # Generate candidate points
            # Mix of random, Sobol, and local perturbations
            n_candidates = 300
            
            # 1. Random points
            cand_random = rng.uniform(-5, 5, (100, dim))
            
            # 2. Sobol points for structured exploration
            sampler_cand = qmc.Sobol(d=dim, scramble=True)
            cand_sobol = sampler_cand.random(100) * 10.0 - 5.0
            
            # 3. Local perturbations around best
            n_local = 100
            # Adaptive local sigma based on length scale
            local_sigma = max(0.1, length_scale * 0.5)
            cand_local = x_best + local_sigma * rng.standard_normal((n_local, dim))
            cand_local = np.clip(cand_local, -5, 5)
            
            candidates = np.vstack([cand_random, cand_sobol, cand_local])
            
            # Evaluate acquisition function
            mean, var = self._gp_predict(candidates, X, y, length_scale, noise, L)
            ei = self._expected_improvement(y_best, mean, var)
            
            # Select best candidate
            best_idx = np.argmax(ei)
            x_new = candidates[best_idx]
            
            # Evaluate function
            f_new = func(x_new)
            
            # Update data
            X = np.vstack([X, x_new])
            y = np.append(y, f_new)
            
            # Update best
            if f_new < y_best:
                y_best = f_new
                x_best = x_new
            
            # Update Cholesky factor incrementally? 
            # For simplicity and robustness, recompute if we just updated hyperparams, 
            # but here we can just recompute every step since N is small (<500).
            # Actually, recomputing Cholesky every step is O(N^3). 
            # With N=500, it's ~0.1s. 500 steps = 50s. Acceptable.
            L = compute_L()

        return x_best, y_best
# EVOLVE-BLOCK-END
