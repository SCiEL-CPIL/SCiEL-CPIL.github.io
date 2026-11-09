import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        
        def check_deadline():
            return time.time() < deadline - 5.0

        # --- Phase 1: Initial Sampling ---
        n_init = 20
        rng = np.random.default_rng(42)
        
        X = []
        Y = []
        
        for i in range(n_init):
            if not check_deadline() or eval_count >= self.budget:
                break
            x = rng.uniform(-5.0, 5.0, self.dim)
            fx = func(x)
            eval_count += 1
            X.append(x)
            Y.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if eval_count >= self.budget or not check_deadline():
            return best_x, best_f

        # --- Phase 2: Surrogate-Assisted Search (GP + EI) ---
        # Hyperparameters
        l = 2.0
        sigma_f = 1.0
        sigma_n = 1e-6
        
        def matern_52_kernel(X1, X2, l, sigma_f):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            r = np.sqrt(dist_sq + 1e-12)
            return sigma_f ** 2 * (1 + np.sqrt(5) * r / l + 5 * r ** 2 / (3 * l ** 2)) * np.exp(-np.sqrt(5) * r / l)

        while eval_count < self.budget and check_deadline():
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            
            # Generate candidates
            n_candidates = 1000
            X_cand = rng.uniform(-5.0, 5.0, (n_candidates, self.dim))
            
            # Compute K and K_s
            K = matern_52_kernel(X_arr, X_arr, l, sigma_f) + sigma_n ** 2 * np.eye(len(X_arr))
            K_s = matern_52_kernel(X_arr, X_cand, l, sigma_f)
            
            try:
                alpha = np.linalg.solve(K, Y_arr)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y_arr, rcond=None)[0]
                
            mean_cand = K_s.T @ alpha
            
            # Variance calculation
            # var(x) = k(x,x) - k(x,X)^T K^-1 k(X,x)
            # We already have K_s = k(X, X_cand). So k(X_cand, X) = K_s.T
            # We need k(X_cand, X_cand) diag
            # To save time, we can approximate var using only the diagonal of K_ss
            # But K_ss is (1000, 1000), computing it is O(1000^2 * 5) which is fine.
            K_ss = matern_52_kernel(X_cand, X_cand, l, sigma_f)
            var_cand = np.diag(K_ss) - np.sum(K_s ** 2, axis=0)
            var_cand = np.maximum(var_cand, 1e-12)
            std_cand = np.sqrt(var_cand)
            
            # EI
            z = (best_f - mean_cand) / std_cand
            phi = 1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * z ** 2)
            Phi = 0.5 * (1 + np.erf(z / np.sqrt(2)))
            ei = (best_f - mean_cand) * Phi + std_cand * phi
            ei = np.where(std_cand < 1e-8, 0.0, ei)
            
            idx_best = np.argmax(ei)
            x_next = X_cand[idx_best]
            
            fx_next = func(x_next)
            eval_count += 1
            
            X.append(x_next)
            Y.append(fx_next)
            
            if fx_next < best_f:
                best_f = fx_next
                best_x = x_next.copy()

        return best_x, best_f
