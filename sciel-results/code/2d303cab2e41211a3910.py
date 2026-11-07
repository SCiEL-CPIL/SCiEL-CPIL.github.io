import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # State
        X = []
        Y = []
        
        # Hyperparameters for GP (Matern 5/2)
        # Length scales: assume domain is [-5, 5], so length scale ~1-2
        # Variance: assume function values are in [-100, 100], so variance ~100-1000
        # We will normalize Y to help GP convergence
        
        # 1. Initial Sampling: Use Halton sequence for deterministic coverage
        def halton(dim, n, skip=0):
            # Simple Halton sequence generator
            def halton_index(i, base):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r
            
            points = np.zeros((n, dim))
            for d in range(dim):
                for i in range(n):
                    points[i, d] = halton_index(i + skip + 1, d + 2) # bases 2, 3, 5...
            return points

        n_init = 20
        X_init = halton(self.dim, n_init)
        # Map [0,1] to [-5, 5]
        X_init = X_init * 10 - 5
        
        for x in X_init:
            y = func(x)
            X.append(x)
            Y.append(y)
            
        X = np.array(X)
        Y = np.array(Y)
        
        # Normalize Y for GP
        Y_mean = np.mean(Y)
        Y_std = np.std(Y)
        if Y_std < 1e-6:
            Y_std = 1.0
        Y_norm = (Y - Y_mean) / Y_std
        
        # GP Parameters
        # Matern 5/2 kernel
        # k(x, x') = (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
        # where d = sqrt(sum((x_i - x'_i)^2 / l_i^2))
        
        # We will fix hyperparameters for speed, or do a very rough optimization.
        # Given the short budget, fixed hyperparameters tuned for [-5,5]^5 and scale ~100 are better.
        # Length scale l = 1.5
        # Signal variance sigma_f = 1.0 (since we normalized Y)
        # Noise variance sigma_n = 0.01
        
        length_scale = 1.5
        sigma_f = 1.0
        sigma_n = 0.01
        
        def matern52_kernel(X1, X2, l):
            # X1: (N, D), X2: (M, D)
            # Returns (N, M)
            sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
            np.maximum(sq_dists, 0, out=sq_dists)
            d = np.sqrt(sq_dists) / l
            return sigma_f**2 * (1 + np.sqrt(5)*d + 5/3*d**2) * np.exp(-np.sqrt(5)*d)

        def gp_predict(X_test, X_train, Y_train, l, sf, sn):
            K = matern52_kernel(X_train, X_train, l) + sn * np.eye(len(X_train))
            K_inv = np.linalg.inv(K)
            K_s = matern52_kernel(X_test, X_train, l)
            
            mean = K_s @ K_inv @ Y_train
            v = K_inv @ K_s.T
            var = sf**2 - np.sum(K_s * v, axis=1)
            var = np.maximum(var, 1e-8)
            return mean, np.sqrt(var)

        def expected_improvement(mean, std, y_best):
            # y_best is the best value in normalized space (lowest)
            # EI = (y_best - mean) * Phi((y_best - mean)/std) + std * phi((y_best - mean)/std)
            # Note: We are minimizing.
            z = (y_best - mean) / std
            from scipy.stats import norm
            ei = (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        # Main Loop
        for _ in range(self.budget - n_init):
            # 1. Predict mean and std on a grid or candidate set
            # To find the next point, we need to maximize EI.
            # We use a multi-start Nelder-Mead on the EI function.
            
            y_best_norm = np.min(Y_norm)
            
            # Generate candidate starting points for acquisition maximization
            # 1. Current best point
            # 2. A few random points
            # 3. Points from a coarse grid
            
            candidates = []
            # Best point
            best_idx = np.argmin(Y_norm)
            candidates.append(X[best_idx])
            
            # Random points
            for _ in range(10):
                candidates.append(np.random.uniform(-5, 5, self.dim))
            
            # Coarse grid points (e.g., 3^5 = 243 points is too many for NM start, take a subset)
            # Let's just use the random ones and the best one.
            
            best_x = None
            best_ei = -np.inf
            
            # Predict on candidates to find a good starting point for NM?
            # Or just run NM from each candidate.
            # NM on EI is expensive if we predict on many points.
            # Let's evaluate EI at the candidates first to pick the top 2-3 for NM.
            
            cand_arr = np.array(candidates)
            means, stds = gp_predict(cand_arr, X, Y_norm, length_scale, sigma_f, sigma_n)
            eis = expected_improvement(means, stds, y_best_norm)
            
            # Pick top 3 candidates for local refinement
            top_indices = np.argsort(eis)[-3:]
            starts = cand_arr[top_indices]
            
            for start in starts:
                # Define acquisition function for NM
                def acq_func(x):
                    x = np.array(x)
                    m, s = gp_predict(x.reshape(1, -1), X, Y_norm, length_scale, sigma_f, sigma_n)
                    return -expected_improvement(m[0], s[0], y_best_norm) # Negate for minimization
                
                res = minimize(acq_func, start, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-5})
                
                if res.fun < best_ei:
                    best_ei = res.fun
                    best_x = res.x
            
            # Evaluate best_x
            x_new = np.clip(best_x, -5, 5)
            y_new = func(x_new)
            
            X = np.vstack([X, x_new])
            Y = np.vstack([Y, y_new])
            
            # Update normalized Y
            Y_mean = np.mean(Y)
            Y_std = np.std(Y)
            if Y_std < 1e-6:
                Y_std = 1.0
            Y_norm = (Y - Y_mean) / Y_std

        return X[np.argmin(Y)], np.min(Y)
