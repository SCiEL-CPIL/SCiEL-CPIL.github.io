"""Bayesian Optimization for BBOB-style continuous functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import t as student_t

class Optimizer:
    """Bayesian Optimizer using Sparse Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        rng = np.random.default_rng(42)
        
        # 1. Initialization: Latin Hypercube Sampling for better coverage
        n_init = 15
        x_init = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            x_init[:, d] = np.sort(rng.random(n_init)) * 10.0 - 5.0
        
        y_init = np.array([func(x) for x in x_init])
        
        X = x_init.copy()
        Y = y_init.copy()
        
        best_x = X[np.argmin(Y)]
        best_y = Y.min()
        
        # Hyperparameters
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-4
        nu = 3.0 # Student-t degrees of freedom
        
        # Acquisition function: Expected Improvement (EI)
        def expected_improvement(x, y_min, y_mean, y_std):
            # Avoid division by zero
            y_std = np.maximum(y_std, 1e-8)
            z = (y_min - y_mean) / y_std
            # For minimization, EI = (y_min - mu) * Phi(z) + sigma * phi(z)
            # Note: Standard EI is for maximization. For minimization:
            # Improvement I(x) = max(0, y_min - f(x))
            # E[I] = (y_min - mu) * Phi((y_min - mu)/sigma) + sigma * phi((y_min - mu)/sigma)
            # Using scipy.stats.norm
            from scipy.stats import norm
            z = (y_min - y_mean) / y_std
            ei = (y_min - y_mean) * norm.cdf(z) + y_std * norm.pdf(z)
            return ei

        # Surrogate model prediction
        def predict_gp(x_new, X_obs, Y_obs):
            """
            Predict mean and variance using a simple GP with Student-t kernel.
            x_new: (n_query, dim)
            X_obs: (n_obs, dim)
            Y_obs: (n_obs,)
            """
            # Normalize Y for numerical stability
            y_mean_obs = np.mean(Y_obs)
            y_std_obs = np.std(Y_obs)
            if y_std_obs < 1e-8:
                y_std_obs = 1.0
            Y_norm = (Y_obs - y_mean_obs) / y_std_obs
            
            # Kernel matrix K between x_new and X_obs
            # k(x, x') = sigma^2 * (1 + ||x-x'||^2 / (2*l^2))^(-nu/2)
            # Using squared euclidean distance
            diff = x_new[:, np.newaxis, :] - X_obs[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            
            K = signal_var * (1 + sq_dist / (2 * length_scale**2)) ** (-nu / 2.0)
            K += noise_var * np.eye(len(X_obs))
            
            # Solve K @ alpha = Y_norm
            try:
                alpha = np.linalg.solve(K, Y_norm)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y_norm, rcond=None)[0]
                
            # Prediction mean
            # mu(x_new) = k(x_new, X_obs) @ alpha * y_std_obs + y_mean_obs
            K_star = signal_var * (1 + sq_dist / (2 * length_scale**2)) ** (-nu / 2.0)
            y_pred = K_star @ alpha * y_std_obs + y_mean_obs
            
            # Prediction variance
            # var(x_new) = k(x_new, x_new) - k(x_new, X_obs) @ K^-1 @ k(X_obs, x_new)
            # We need K_inv or use Cholesky. Let's assume K is SPD.
            try:
                L = np.linalg.cholesky(K)
                K_inv_Y = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                # var = signal_var - K_star @ K_inv_K_star
                # K_star @ K_inv @ K_star.T
                # Note: K_inv = L^-T @ L^-1
                # K_star @ L^-T @ L^-1 @ K_star.T
                temp = np.linalg.solve(L.T, K_star.T)
                var = signal_var - np.sum(K_star * temp.T, axis=1)
            except np.linalg.LinAlgError:
                # Fallback to simple variance if Cholesky fails
                var = np.full(len(x_new), signal_var)
                
            var = np.maximum(var, 1e-8)
            y_std = np.sqrt(var) * y_std_obs
            
            return y_pred, y_std

        # Main loop
        calls_made = n_init
        for i in range(self.budget - n_init):
            if time.time() > deadline:
                break
                
            # 1. Generate candidate points for acquisition
            # Mix of random points and points near current best
            n_cands = 200
            x_cands = rng.uniform(-5, 5, (n_cands, self.dim))
            
            # Predict
            y_mean, y_std = predict_gp(x_cands, X, Y)
            
            # 2. Calculate EI
            # Add small jitter to encourage exploration if std is high? 
            # EI naturally handles this.
            ei = expected_improvement(x_cands, best_y, y_mean, y_std)
            
            # 3. Select best candidate
            idx = np.argmax(ei)
            x_next = x_cands[idx]
            
            # 4. Evaluate
            y_next = func(x_next)
            calls_made += 1
            
            # 5. Update
            X = np.vstack([X, x_next])
            Y = np.append(Y, y_next)
            
            if y_next < best_y:
                best_y = y_next
                best_x = x_next
            
            # Adaptive hyperparameters (very simple)
            # If we are stuck, increase length scale to explore more broadly
            # If we are improving rapidly, decrease length scale to exploit
            # For simplicity, keep fixed for now, but we could add a simple rule.
            # Let's add a simple stagnation check.
            # If no improvement in last 10 calls, increase length_scale
            # This requires tracking history, which we don't have easily here without extra state.
            # We'll stick to fixed hyperparameters for stability in this implementation.
            
        return best_x, best_y

# EVOLVE-BLOCK-END
