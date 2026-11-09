"""Surrogate-assisted optimizer using Gaussian Process + Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import Sobol

class Optimizer:
    """GP surrogate with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # 1. Initialization with Sobol sequence for better space-filling
        n_init = min(20, budget)
        sampler = Sobol(d=dim, scramble=True)
        X_raw = sampler.random(n_init) * 10 - 5  # Scale to [-5, 5]
        Y_raw = np.array([func(x) for x in X_raw])
        
        best_idx = np.argmin(Y_raw)
        x_best = X_raw[best_idx].copy()
        f_best = Y_raw[best_idx]
        
        # State for GP
        X_train = X_raw.copy()
        Y_train = Y_raw.copy()
        
        # Hyperparameters
        # We will normalize X to [0, 1] and Y to have zero mean and unit variance for GP stability
        # But we need to keep track of original values for evaluation
        
        def normalize_X(X):
            return (X + 5.0) / 10.0 # Map [-5, 5] to [0, 1]
            
        def denormalize_X(Xn):
            return Xn * 10.0 - 5.0 # Map [0, 1] to [-5, 5]

        # GP Hyperparameters
        length_scale = 0.2 # Initial guess in normalized space
        signal_var = 1.0
        noise_var = 1e-2
        
        def kernel(A, B, ls):
            """Squared exponential kernel."""
            diff = A[:, None, :] - B[None, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            return signal_var * np.exp(-sq_dist / (2 * ls ** 2))

        def gp_predict(X_new_norm, X_train_norm, Y_train_norm, ls):
            """Predict mean and variance at X_new given training data."""
            K = kernel(X_train_norm, X_train_norm, ls)
            K += noise_var * np.eye(len(X_train_norm))
            K_new = kernel(X_train_norm, X_new_norm, ls)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_norm))
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(len(X_train_norm))
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_norm))
            
            mean_norm = K_new.T @ alpha
            v = np.linalg.solve(L, K_new.T)
            var_norm = signal_var - np.sum(v ** 2, axis=0)
            var_norm = np.maximum(var_norm, 1e-10)
            
            return mean_norm, var_norm

        def log_likelihood(ls):
            """Calculate log-likelihood for a given length scale."""
            X_train_norm = normalize_X(X_train)
            y_mean = np.mean(Y_train)
            y_std = np.std(Y_train)
            if y_std < 1e-8:
                y_std = 1.0
            Y_train_norm = (Y_train - y_mean) / y_std
            
            K = kernel(X_train_norm, X_train_norm, ls)
            K += noise_var * np.eye(len(X_train_norm))
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_norm))
                ll = -0.5 * np.dot(Y_train_norm, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * len(Y_train) * np.log(2 * np.pi)
                return ll
            except np.linalg.LinAlgError:
                return -np.inf

        def expected_improvement(x_norm, f_best, X_train_norm, Y_train_norm, ls):
            """Compute EI for a single point x_norm."""
            x = np.array([x_norm])
            mean_norm, var_norm = gp_predict(x, X_train_norm, Y_train_norm, ls)
            sigma = np.sqrt(var_norm[0])
            if sigma < 1e-8:
                return 0.0
            z = (f_best_norm - mean_norm[0]) / sigma
            ei = (f_best_norm - mean_norm[0]) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei  # Negative for minimization

        # Main optimization loop
        calls_made = n_init
        # Precompute normalized training data
        X_train_norm = normalize_X(X_train)
        y_mean = np.mean(Y_train)
        y_std = np.std(Y_train)
        if y_std < 1e-8:
            y_std = 1.0
        Y_train_norm = (Y_train - y_mean) / y_std
        f_best_norm = (f_best - y_mean) / y_std
        
        for i in range(budget - n_init):
            # Periodically re-estimate length scale
            if i % 10 == 0 or i == 0:
                # Grid search for length scale
                candidate_ls = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
                best_ll = -np.inf
                best_ls = length_scale
                for ls in candidate_ls:
                    ll = log_likelihood(ls)
                    if ll > best_ll:
                        best_ll = ll
                        best_ls = ls
                length_scale = best_ls
                
            # Update normalized best
            f_best_norm = (f_best - y_mean) / y_std
            
            # Find next point using EI
            best_ei = -np.inf
            x_next_norm = None
            
            # Start points for L-BFGS-B
            n_restarts = 10
            starts = []
            # 1. Best point
            starts.append(normalize_X(x_best))
            # 2. Centroid
            starts.append(np.mean(X_train_norm, axis=0))
            # 3. Random points
            for _ in range(n_restarts - 2):
                starts.append(rng.uniform(0, 1, dim))
            
            for x0 in starts:
                x0 = np.clip(x0, 0, 1)
                try:
                    res = minimize(
                        lambda x: expected_improvement(x, f_best, X_train_norm, Y_train_norm, length_scale),
                        x0,
                        bounds=[(0, 1)] * dim,
                        method='L-BFGS-B',
                        options={'maxiter': 50}
                    )
                    if res.fun < best_ei:
                        best_ei = res.fun
                        x_next_norm = np.clip(res.x, 0, 1)
                except Exception:
                    continue
            
            if x_next_norm is None:
                x_next_norm = rng.uniform(0, 1, dim)
            
            # Map back to original space
            x_next = denormalize_X(x_next_norm)
            
            # Evaluate the new point
            f_new = func(x_next)
            
            # Update training data
            X_train = np.vstack([X_train, x_next])
            Y_train = np.append(Y_train, f_new)
            
            # Update normalization
            y_mean = np.mean(Y_train)
            y_std = np.std(Y_train)
            if y_std < 1e-8:
                y_std = 1.0
            X_train_norm = normalize_X(X_train)
            Y_train_norm = (Y_train - y_mean) / y_std
            
            calls_made += 1
            
            # Update best
            if f_new < f_best:
                f_best = f_new
                x_best = x_next.copy()
        
        return x_best, f_best
# EVOLVE-BLOCK-END
