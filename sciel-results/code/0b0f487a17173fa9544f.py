import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        
        # GP Parameters
        # Matern kernel is more flexible than RBF for non-smooth landscapes, 
        # but Schaffers F7 is continuous. Let's use Matern with nu=2.5 or RBF.
        # RBF is computationally cheaper and often sufficient for smooth-ish BBOB functions.
        # Let's use RBF (Squared Exponential) with automatic length scales.
        
        # State
        X = np.zeros((0, dim))
        Y = np.zeros(0)
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Hyperparameters for GP
        # Initial length scales: assume function varies on scale of ~2.0
        # We will optimize these roughly or keep them fixed for speed.
        # Fixed length scales are faster. Let's start with uniform length scale 1.5.
        # Signal variance sigma_f^2, noise variance sigma_n^2
        sigma_f2 = 1.0
        sigma_n2 = 1e-8
        
        # Kernel function: Matern 5/2 is robust. 
        # k(x, y) = sigma_f^2 * (1 + d/sqrt(5)*l + (d^2)/(5*l^2)) * exp(-d/sqrt(5)*l)
        # where d is Euclidean distance, l is length scale.
        # For simplicity and speed in dim=5, let's use RBF:
        # k(x, y) = sigma_f^2 * exp(-0.5 * ||x-y||^2 / l^2)
        
        # Adaptive length scale? 
        # Let's start with l=1.0 and maybe adapt it if we have time, but fixed is safer for budget.
        # Actually, the domain is [-5,5]^5. A length scale of 1.0 is reasonable.
        
        length_scale = 1.0
        
        def kernel(X1, X2, ls, sf2):
            # X1: (n1, dim), X2: (n2, dim)
            # Compute squared distances
            # Use broadcasting: X1[:, None, :] - X2[None, :, :]
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            return sf2 * np.exp(-0.5 * sq_dists / (ls**2))
        
        def track_call(x):
            nonlocal calls_used
            val = func(x)
            calls_used += 1
            return val
        
        # 1. Initial Exploration
        # Sample 10 points uniformly in [-5, 5]^5
        n_init = 10
        X_init = np.random.uniform(-5, 5, size=(n_init, dim))
        Y_init = np.array([track_call(x) for x in X_init])
        
        X = X_init
        Y = Y_init
        
        # Find initial best
        idx_best = np.argmin(Y)
        best_f = Y[idx_best]
        best_x = X[idx_best].copy()
        
        # 2. Sequential Optimization Loop
        # Use Expected Improvement (EI) acquisition function
        # EI(x) = (mu_best - mu(x) - xi) * Phi((mu_best - mu(x) - xi) / sigma(x)) + sigma(x) * phi((mu_best - mu(x) - xi) / sigma(x))
        # where mu(x), sigma(x) are GP mean and std dev at x.
        # xi is exploration-exploitation trade-off parameter.
        
        xi = 0.01 # Small value to encourage exploitation
        
        # Candidate points for EI optimization
        # Since dim=5, we can use a random restart local search for EI maximization.
        # Or simply sample many random points and pick the best EI.
        # Given budget is 500, and GP update is O(n^3), we need to be careful.
        # n grows up to 500. 500^3 = 125M operations per update. 
        # In Python/NumPy, this might be slow. 
        # Let's check speed. 500x500 matrix inversion is fast. 
        # But we do this 490 times. 490 * (500^3) is too slow if done naively in pure Python loops for kernel evals.
        # However, kernel evals are vectorized. 
        # The bottleneck is Cholesky decomposition of K + sigma_n^2 I.
        # For n=500, dim=5, this is very fast in NumPy/SciPy.
        
        # To speed up EI maximization, we will:
        # 1. Generate a large set of random candidate points (e.g., 2000).
        # 2. Compute GP mean and variance at these points.
        # 3. Compute EI.
        # 4. Pick the best.
        # This avoids expensive local optimization of EI.
        
        n_candidates = 2000
        
        while calls_used < budget:
            if time.time() > deadline:
                break
                
            # Update GP
            # K = kernel(X, X) + sigma_n^2 * I
            K = kernel(X, X, length_scale, sigma_f2)
            K += sigma_n2 * np.eye(len(X))
            
            # Cholesky decomposition
            try:
                L_k = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Regularize if not positive definite
                K += 1e-6 * np.eye(len(X))
                L_k = np.linalg.cholesky(K)
            
            # Solve for alpha: K * alpha = Y
            # alpha = L^{-T} (L^{-1} Y)
            v = np.linalg.solve(L_k, Y)
            alpha = np.linalg.solve(L_k.T, v)
            
            # Generate candidate points
            X_cand = np.random.uniform(-5, 5, size=(n_candidates, dim))
            
            # Compute GP mean and variance at candidates
            # K_s = kernel(X_cand, X)
            K_s = kernel(X_cand, X, length_scale, sigma_f2)
            
            # Mean: mu = K_s @ alpha
            mu = K_s @ alpha
            
            # Variance: sigma^2 = k(x, x) - K_s @ K^{-1} @ K_s.T
            # We can compute this as: k(x,x) - K_s @ (K^{-1} K_s.T)
            # Or use Cholesky: sigma^2 = k(x,x) - sum((L^{-1} K_s.T)^2, axis=1)
            # Let K_s.T be (n, m). L^{-1} K_s.T is (m, m) if L is (m,m) and K_s.T is (m, m)? No.
            # K_s is (n_cand, n_data). K_s.T is (n_data, n_cand).
            # We need to solve L @ Z = K_s.T. Z is (n_data, n_cand).
            # Then variance = k(x,x) - sum(Z^2, axis=0)
            
            Z = np.linalg.solve(L_k, K_s.T)
            var = sigma_f2 - np.sum(Z**2, axis=0)
            # Clip variance to be non-negative
            var = np.maximum(var, 1e-8)
            std = np.sqrt(var)
            
            # Compute EI
            # best_f is the current minimum
            # EI = (best_f - mu - xi) * Phi((best_f - mu - xi) / std) + std * phi((best_f - mu - xi) / std)
            
            Z_ei = (best_f - mu - xi) / std
            EI = (best_f - mu - xi) * norm.cdf(Z_ei) + std * norm.pdf(Z_ei)
            
            # Select candidate with max EI
            idx_best_cand = np.argmax(EI)
            x_new = X_cand[idx_best_cand]
            
            # Evaluate
            f_new = track_call(x_new)
            
            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                
            # Add to GP data
            X = np.vstack([X, x_new])
            Y = np.append(Y, f_new)
            
            # Optional: Adapt length scale?
            # If we have time, we could re-estimate hyperparameters.
            # But for speed and stability, fixed length scale is a good first try.
            # If the function is very flat or very steep, it might fail.
            # Schaffers F7 is moderately smooth.
            
        return best_x, best_f
