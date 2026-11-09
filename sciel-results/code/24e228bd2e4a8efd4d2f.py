"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_val = np.inf
        best_x = np.zeros(dim)

        def call_func(x):
            nonlocal calls_used, best_val, best_x
            if calls_used >= budget:
                return best_val
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            calls_used += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
            return val

        # CMA-ES Parameters
        n = dim
        # Population size
        lam = 4 + int(3 * np.log(n))
        if lam < 4:
            lam = 4
        # Number of parents
        mu = max(1, lam // 2)
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)
        
        # Step size
        sigma = 2.0
        # Mean
        mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        # Inverse square root of C
        # We need to compute C^{-1/2} for sampling
        # Use eigendecomposition for stability in low dimensions
        def invsqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        Cinv_sqrt = invsqrt(C)
        
        # Path accumulators
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + np.log(n) / n) / (n + 4)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        # Normalization of step size
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Initial evaluation
        f_mean = call_func(mean)
        
        while calls_used < budget and time.time() < deadline:
            # Sample lambda points
            ys = []
            fs = []
            for i in range(lam):
                if calls_used >= budget:
                    break
                # Standard normal
                z = np.random.standard_normal(n)
                # Transform to x
                x = mean + sigma * (Cinv_sqrt @ z)
                x = np.clip(x, -5.0, 5.0)
                fx = call_func(x)
                ys.append(x)
                fs.append(fx)
            
            if not ys:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = np.array(ys)[idx]
            fs_sorted = np.array(fs)[idx]
            
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * ys_sorted[:mu], axis=0)
            
            # Update step size
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma)mueff) / sigma * (Cinv_sqrt @ (new_mean - mean))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (Cinv_sqrt @ (new_mean - mean)) / sigma
            
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used // lam + 1))) / chi_n
            h_sigma = min(1.0, h_sigma)
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (new_mean - mean) / sigma
            
            # Update covariance
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma^2) c_c (2 - c_c) C) + c_mu sum w_i (y_i - mean)(y_i - mean)^T
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(ys_sorted[:mu] - mean, ys_sorted[:mu] - mean), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - h_sigma ** 2) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2.0
            
            # Update Cinv_sqrt
            Cinv_sqrt = invsqrt(C)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            mean = new_mean
            mean = np.clip(mean, -5.0, 5.0)
            
            # Update best
            # The best in the population is ys_sorted[0]
            if fs_sorted[0] < best_val:
                best_val = fs_sorted[0]
                best_x = ys_sorted[0].copy()

        return best_x, best_val
# EVOLVE-BLOCK-END
