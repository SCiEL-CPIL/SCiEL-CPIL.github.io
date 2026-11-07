"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimizer with Gaussian Process surrogate for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 280.0  # Safety margin

        # Initialize with random points to build initial surrogate
        # Use a fixed seed for reproducibility within the instance if needed, 
        # but random is fine. Let's use a standard RNG.
        rng = np.random.default_rng(42)
        
        # Initial population size: enough to fit a GP reasonably well in 5D
        # 5D is low, so 20-50 points is a good start.
        n_init = 30
        if n_init > budget:
            n_init = budget
            
        X = rng.uniform(-5, 5, size=(n_init, dim))
        Y = np.array([func(x) for x in X])
        
        calls = n_init
        best_x = X[np.argmin(Y)].copy()
        best_y = Y.min()
        
        # GP Parameters
        # We will use a simple RBF kernel with automatic hyperparameter optimization
        # or fixed reasonable hyperparameters to save time.
        # Given the 300s limit and 500 calls, we can afford some computation per step.
        # However, fitting a GP with 500 points in 5D is feasible but might be slow if we re-optimize hyperparams every step.
        # Strategy: Use a fixed length scale or optimize once and update, or use a simpler kernel.
        # Let's use a standard RBF kernel.
        
        # Hyperparameters: length_scale, noise, signal_variance
        # We can optimize these occasionally or use a robust default.
        # For Lunacek (bimodal, rotated), a moderate length scale is needed.
        
        # To be safe and fast, we will use a simple approach:
        # 1. Standardize inputs and outputs? 
        #    Standardizing inputs helps GP convergence.
        #    Standardizing outputs helps EI calculation.
        
        # Let's implement a lightweight GP fit.
        # Since dim=5 and N<=500, Cholesky decomposition is O(N^3) ~ 125M ops max.
        # Python/NumPy can handle this, but 500 iterations * fit time might be tight.
        # Alternative: Use a simpler surrogate like Random Forest or KNN?
        # GP is better for smooth-ish functions, but Lunacek has oscillations (Rastrigin part).
        # However, the "Lunacek bi-Rastrigin" is often smoother than pure Rastrigin in the basin.
        
        # Let's try a GP with fixed hyperparameters first to ensure speed, 
        # or optimize hyperparams every 10 steps.
        
        # Actually, a simpler and very effective method for low-dim black box is 
        # "Sequential Model-Based Optimization" with a GP.
        
        # To avoid the cost of hyperparameter optimization every step, 
        # we will optimize hyperparameters every 10 iterations or when the model likelihood drops.
        
        # Let's define the GP model manually using NumPy for speed and control.
        
        def rbf_kernel(X1, X2, l):
            # X1: (N, D), X2: (M, D)
            # Distance matrix
            D = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            D = np.maximum(D, 0)
            return np.exp(-0.5 * D / (l * l))
        
        # State variables
        # We'll keep X, Y
        # Hyperparameters: l (length scale), sigma_f (signal var), sigma_n (noise)
        # Let's initialize l=1.0, sigma_f=1.0, sigma_n=0.01
        
        l = 1.0
        sigma_f = 1.0
        sigma_n = 0.01
        
        # For EI, we need the mean and variance of the GP.
        
        # Optimization loop
        while calls < budget:
            if time.time() - start_time > max_time:
                break
            
            # Every 10 steps, re-optimize hyperparameters
            if calls % 10 == 0 and calls > n_init:
                # Optimize l, sigma_f, sigma_n via a simple grid or local search
                # To save time, let's just do a few random restarts or a simple gradient step?
                # Actually, for robustness and speed, let's stick to fixed or slowly adapting hyperparams.
                # A common trick: adapt l based on the scale of improvements.
                # Or just use a fixed l=1.0 and let the noise handle the rest.
                # Given the complexity, let's try to optimize l in a small grid every 10 steps.
                
                best_l = 1.0
                best_lik = -np.inf
                # Grid search for l in log scale
                for l_candidate in [0.2, 0.5, 1.0, 2.0, 5.0]:
                    K = rbf_kernel(X, X, l_candidate)
                    K += (sigma_n + 1e-8) * np.eye(len(X))
                    try:
                        L = np.linalg.cholesky(K)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
                        lik = -0.5 * Y @ alpha - 0.5 * np.sum(np.log(np.diag(L))) - 0.5 * len(Y) * np.log(2 * np.pi)
                        if lik > best_lik:
                            best_lik = lik
                            best_l = l_candidate
                    except:
                        continue
                l = best_l
                
            # Generate candidate points for EI
            # We need to sample many candidates and compute EI for each.
            # This is the expensive part if we do it naively for every step.
            # N candidates = 200?
            # GP prediction for N candidates at M points is O(N*M*D + N*M^2).
            # With M=500, N=200, D=5.
            # 200*500*5 = 500k mults. 200*500*500 = 50M mults for K(X_new, X_old).
            # This is doable in Python if vectorized.
            
            n_candidates = 100
            X_cand = rng.uniform(-5, 5, size=(n_candidates, dim))
            
            # Compute GP mean and variance for candidates
            # K(X_cand, X)
            K_cand = rbf_kernel(X_cand, X, l)
            K = rbf_kernel(X, X, l)
            K += (sigma_n + 1e-8) * np.eye(len(X))
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
                v = np.linalg.solve(L, K_cand.T)
                
                # Mean
                mu_cand = K_cand @ alpha
                # Variance
                var_cand = sigma_f**2 - np.sum(v**2, axis=1)
                var_cand = np.maximum(var_cand, 1e-8)
                std_cand = np.sqrt(var_cand)
                
                # Expected Improvement
                # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
                # Note: f_best is the minimum so far.
                # Standard EI for minimization:
                # Improvement = f_best - f(x)
                # E[Improvement] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
                
                z = (best_y - mu_cand) / std_cand
                # Clip z to avoid numerical issues
                z = np.clip(z, -10, 10)
                
                ei = (best_y - mu_cand) * norm.cdf(z) + std_cand * norm.pdf(z)
                
                # Select best candidate
                idx = np.argmax(ei)
                x_new = X_cand[idx]
                
            except:
                # Fallback to random
                x_new = rng.uniform(-5, 5, size=dim)
            
            # Evaluate
            y_new = func(x_new)
            calls += 1
            
            # Update data
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()
                
        return best_x, best_y

# EVOLVE-BLOCK-END
