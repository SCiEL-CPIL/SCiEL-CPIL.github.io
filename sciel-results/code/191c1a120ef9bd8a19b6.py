"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Surrogate-Assisted Bayesian Optimization with GP and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        # Initialize with a small random sample to start the surrogate
        rng = np.random.default_rng(42)
        
        X = []
        Y = []
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Phase 1: Initial random sampling to get a baseline for the GP
        # 50 points is usually enough to start a GP in 5D without being too expensive
        n_init = min(50, self.budget)
        
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            
            X.append(x)
            Y.append(fx)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            if calls_made >= self.budget:
                break
                
        if not X:
            # Fallback if no calls were made (shouldn't happen)
            x = np.zeros(self.dim)
            fx = func(x)
            calls_made = 1
            return x, fx

        X = np.array(X)
        Y = np.array(Y)
        
        # Helper to compute Expected Improvement
        def expected_improvement(x, y_pred, y_std):
            # EI = (mu - best - xi) * Phi(Z) + sigma * phi(Z)
            # where Z = (mu - best - xi) / sigma
            xi = 0.01 * (best_f - np.min(Y)) # small exploitation bias
            if y_std < 1e-10:
                return 0.0
            Z = (y_pred - best_f - xi) / y_std
            # Standard Normal PDF and CDF
            phi = np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)
            Phi = 0.5 * (1 + np.vectorize(lambda z: 1 if z > 0 else -1)(Z) * 0) # Placeholder, use scipy
            from scipy.stats import norm
            Phi = norm.cdf(Z)
            return (y_pred - best_f - xi) * Phi + y_std * phi

        # Phase 2: Iterative Surrogate Optimization
        # We will use a simple Matern kernel or RBF. 
        # To keep it fast and robust without sklearn, we implement a simple GP update 
        # or use a simpler surrogate like a Radial Basis Function (RBF) interpolant 
        # if scipy GP is too slow. However, scipy has no built-in GP regression 
        # that is super fast for 5D. 
        # Let's use a simpler approach: Kriging via scipy.stats or manual GP.
        # Actually, implementing a full GP in pure Python/NumPy for 5D with 500 points 
        # might be slow if we re-fit every step. 
        # Alternative: Use a "Gaussian Process" approximation or just a strong 
        # Local Search around the best point found by the GP suggestion.
        
        # Let's try a hybrid: 
        # 1. Fit a simple RBF surrogate (fast)
        # 2. Optimize EI on the surrogate
        # 3. Evaluate the true function
        # 4. Repeat
        
        # To avoid complex GP math, we can use a simpler surrogate: 
        # A local quadratic model or just use the structure of the problem.
        # But the directive asks for GP or RF. 
        # Let's implement a lightweight GP using NumPy.
        
        def matern_kernel(X1, X2, length_scale=1.0, variance=1.0):
            # Simplified Matern 5/2 or just RBF for speed
            # RBF is faster
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            return variance * np.exp(-0.5 * dist_sq / (length_scale**2))

        # We will use a simplified iterative process:
        # 1. Add noise to prevent singularity
        # 2. Compute K matrix
        # 3. Compute alpha
        # 4. Predict mean and var for candidate points
        # 5. Maximize EI
        
        # Since fitting a GP from scratch every iteration is O(N^3), and N grows to 500,
        # this might be too slow in Python if not optimized. 
        # 500^3 is 125M operations, which is borderline in 300s if done 450 times.
        # We need to be careful. 
        # Strategy: 
        # - Use a smaller subset for the GP if it gets too slow? 
        # - Or use a simpler surrogate like a Linear Model + Noise?
        # - Or use a "Gaussian Process" only for the first 100 points, then switch to CMA-ES?
        # The directive says "Surrogate-assisted... GP or random-forest".
        # Let's stick to a simplified GP but optimize the loop.
        # We can use Cholesky decomposition for stability and speed.
        
        from scipy.linalg import cho_factor, cho_solve, cholesky
        
        length_scale = 1.5
        noise_var = 1e-6
        signal_var = 1.0
        
        while calls_made < self.budget and time.time() < deadline:
            # 1. Fit GP
            K = matern_kernel(X, X, length_scale, signal_var)
            K += noise_var * np.eye(len(X))
            
            try:
                # Cholesky factorization
                L = cholesky(K, lower=True)
                # Solve L alpha = Y
                alpha = cho_solve((L, True), Y)
            except Exception:
                # If Cholesky fails, add more noise and retry once
                K += 1e-4 * np.eye(len(X))
                try:
                    L = cholesky(K, lower=True)
                    alpha = cho_solve((L, True), Y)
                except Exception:
                    # Fallback to random search if GP fails
                    x = rng.uniform(-5, 5, self.dim)
                    fx = func(x)
                    calls_made += 1
                    X = np.vstack([X, x])
                    Y = np.append(Y, fx)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    continue

            # 2. Generate candidate points for EI optimization
            # We sample a grid or use random candidates to find the max EI
            # To be efficient, we sample 200 random candidates and pick the best EI
            n_candidates = 200
            candidates = rng.uniform(-5, 5, (n_candidates, self.dim))
            
            # Compute K(X_new, X_old)
            K_star = matern_kernel(candidates, X, length_scale, signal_var)
            
            # Predictions: mu = K_star @ alpha
            mu = K_star @ alpha
            
            # Variance: var = k(x_new, x_new) - K_star @ K^{-1} @ K_star.T
            # k(x_new, x_new) is signal_var for all
            v = cho_solve((L, True), K_star.T) # L v = K_star.T
            var = signal_var - np.sum(K_star * v, axis=1)
            var = np.maximum(var, 1e-8) # numerical stability
            std = np.sqrt(var)
            
            # Compute EI
            # best_f is the current global minimum
            # EI formula: (mu - best_f) * Phi(Z) + std * phi(Z)
            Z = (mu - best_f) / std
            from scipy.stats import norm
            EI = (mu - best_f) * norm.cdf(Z) + std * norm.pdf(Z)
            
            # Select candidate with max EI
            idx = np.argmax(EI)
            x_next = candidates[idx]
            
            # 3. Evaluate
            fx = func(x_next)
            calls_made += 1
            
            X = np.vstack([X, x_next])
            Y = np.append(Y, fx)
            
            if fx < best_f:
                best_f = fx
                best_x = x_next.copy()
                
            # Adaptive length scale? 
            # If we are stagnating, increase exploration (increase length scale or noise)
            # For simplicity, we keep it fixed, but we could adjust.
            
            # Check time
            if time.time() > deadline:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
