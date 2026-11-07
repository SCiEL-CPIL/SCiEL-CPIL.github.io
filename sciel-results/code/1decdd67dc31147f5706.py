"""Optimizer for BBOB-style f17 (Schaffers F7) using Bayesian Optimization with GP."""
import numpy as np
import time
from scipy.stats import norm
import math

class Optimizer:
    """Bayesian Optimizer using Gaussian Process surrogate and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: kill at 280s, limit is 300s (evaluator) / 330s (hard kill)
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        # Hyperparameters for GP
        # Length scales: Initial guess. Since box is [-5,5], scale 2.0 is reasonable.
        # We will adapt these if needed, but for a 500 budget, fixed is often sufficient 
        # if well-chosen. Schaffers F7 has a specific structure.
        # f17 = (sin^2(x1) + sin^2(x2) ... ) / 100 + sum((xi+1)^2)/100
        # The sin^2 part oscillates with period 2*pi ~ 6.28. 
        # The quadratic part is smooth.
        # Let's use a Matern 5/2 kernel which is flexible.
        
        # Initial exploration
        # Use a small Sobol or random set to initialize the GP
        n_init = 10
        X_train = np.random.uniform(-5, 5, size=(n_init, dim))
        y_train = np.zeros(n_init)
        
        count = 0
        best_x = X_train[0].copy()
        best_f = float('inf')
        
        # Evaluate initial points
        for i in range(n_init):
            if time.time() > deadline:
                break
            try:
                f = func(X_train[i])
                count += 1
                if f < best_f:
                    best_f = f
                    best_x = X_train[i].copy()
                y_train[i] = f
            except Exception:
                # If evaluation fails, stop
                break
                
        if count >= budget:
            return best_x, best_f

        # GP Parameters
        # Noise variance (observation noise)
        noise_var = 1e-10
        # Signal variance
        signal_var = 1.0
        # Length scales (per dimension)
        # Initialize to 2.0 (covers a good chunk of the [-5,5] range)
        length_scales = np.ones(dim) * 2.0
        
        # Kernel function: Matern 5/2
        def matern52_kernel(X1, X2, ls):
            # X1: (n1, d), X2: (n2, d)
            # Compute scaled distance
            X1_scaled = X1 / ls
            X2_scaled = X2 / ls
            # Euclidean distance
            diff = X1_scaled[:, np.newaxis, :] - X2_scaled[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            # Matern 5/2 kernel
            sqrt5 = np.sqrt(5.0)
            k = np.exp(-sqrt5 * dist) * (1 + sqrt5 * dist + 5.0/3.0 * dist**2)
            return k

        # Acquisition Function: Expected Improvement
        def expected_improvement(X_pred, mean, std, best_f):
            # X_pred: (n, d)
            # mean: (n,)
            # std: (n,)
            # best_f: float
            imp = best_f - mean
            # Avoid division by zero
            std_safe = np.maximum(std, 1e-10)
            z = imp / std_safe
            ei = imp * norm.cdf(z) + std_safe * norm.pdf(z)
            # Where imp <= 0, EI should be 0
            ei[imp <= 0] = 0.0
            return ei

        # Main loop
        while count < budget:
            if time.time() > deadline:
                break
            
            # 1. Compute GP predictions on a candidate set
            # We need to generate candidates. 
            # Strategy: Use a combination of random sampling and local refinement around best points.
            # For 500 budget, generating a large candidate set every step is expensive.
            # Let's use a fixed number of candidates, e.g., 2000.
            
            # Generate candidates
            # 80% random, 20% local around current best
            n_cand = 2000
            X_cand = np.random.uniform(-5, 5, size=(n_cand, dim))
            n_local = int(0.2 * n_cand)
            # Local candidates: best_x + small noise
            local_scale = 0.5
            X_local = best_x + np.random.normal(0, local_scale, size=(n_local, dim))
            X_local = np.clip(X_local, -5, 5)
            X_cand = np.vstack([X_cand[:n_cand - n_local], X_local])
            
            # 2. Compute K(X_train, X_train) + noise * I
            K_train = matern52_kernel(X_train, X_train, length_scales)
            K_train += noise_var * np.eye(len(X_train))
            
            # 3. Compute K(X_train, X_cand)
            K_test = matern52_kernel(X_train, X_cand, length_scales)
            
            # 4. Compute mean and variance
            try:
                # Solve K_train * alpha = y_train
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                
                # Mean: K_test.T * alpha
                mean = K_test.T @ alpha
                
                # Variance: k(X_cand, X_cand) - K_test.T * inv(K_train) * K_test
                # We can compute this as k0 - K_test.T @ (L^-T @ L^-1 @ K_test)
                # Let v = L^-1 @ K_test
                v = np.linalg.solve(L, K_test)
                var = signal_var - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-10) # Numerical stability
                std = np.sqrt(var)
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails (should be rare with noise)
                # Use pseudo-inverse
                try:
                    K_inv = np.linalg.pinv(K_train)
                    mean = K_test.T @ (K_inv @ y_train)
                    var = signal_var - np.sum((K_test.T @ K_inv) * K_test, axis=0)
                    var = np.maximum(var, 1e-10)
                    std = np.sqrt(var)
                except Exception:
                    # If GP fails completely, fall back to random search
                    X_next = np.random.uniform(-5, 5, size=(1, dim))
                    try:
                        f = func(X_next[0])
                        count += 1
                        if f < best_f:
                            best_f = f
                            best_x = X_next[0].copy()
                        X_train = np.vstack([X_train, X_next])
                        y_train = np.append(y_train, f)
                    except Exception:
                        break
                    continue

            # 5. Select next point using EI
            ei = expected_improvement(X_cand, mean, std, best_f)
            idx = np.argmax(ei)
            X_next = X_cand[idx].reshape(1, -1)
            
            # 6. Evaluate
            try:
                f = func(X_next[0])
                count += 1
                if f < best_f:
                    best_f = f
                    best_x = X_next[0].copy()
                
                # 7. Update training data
                X_train = np.vstack([X_train, X_next])
                y_train = np.append(y_train, f)
                
            except Exception:
                break

        return best_x, best_f
