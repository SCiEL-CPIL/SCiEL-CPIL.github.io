"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s kill

        # CMA-ES Parameters
        # Initial point: center of the box
        x = np.zeros(dim)
        f_x = func(x)
        
        # Initial sigma
        # Heuristic for box size 10: sigma0 = 2.0 is reasonable
        sigma = 2.0
        
        # Population size
        # lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(dim)) if dim > 1 else 4
        
        # Weights
        mu = lam // 2  # top half
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Learning rates
        cc = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        cs = (lam / 2 + 1) / (dim + lam / 2 + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + lam)
        cmu = min(1 - c1, 2 * ((dim - 1.3) / (dim + 1) ** 2 - 1))
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        # Inverse square root of C
        D = np.sqrt(np.diag(C))
        B = np.eye(dim)
        
        # Eigen decomposition helper
        def update_BD(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-10)
            D = np.sqrt(eigvals)
            B = eigvecs
            return D, B

        D, B = update_BD(C)
        
        # Best so far
        best_x = x.copy()
        best_f = f_x
        
        calls = 1
        gen = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            ys = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                # Clip to box [-5, 5]
                y = np.clip(y, -5, 5)
                ys[i] = y
            
            # Evaluate
            fs = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    # Fill remaining with best known if we run out of time
                    fs[i] = best_f
                    continue
                fs[i] = func(ys[i])
                calls += 1
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = ys[i].copy()
            
            # Sort
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            
            # Update mean
            # Re-weighted mean
            w = weights
            x_new = np.sum(ys_sorted[:mu] * w[:, None], axis=0)
            
            # Cumulation
            y_sigma = (x_new - x) / (sigma * D)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * y_sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1)) <
                    (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * y_sigma
            
            # Update covariance
            delta = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            C += cmu * np.sum(w[i] * np.outer(ys_sorted[i] - x, ys_sorted[i] - x) for i in range(mu))
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update step size (1/5 rule equivalent in CMA)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update B and D
            D, B = update_BD(C)
            
            x = x_new
            gen += 1
            
            # If sigma becomes too small or too large, break
            if sigma < 1e-10 or sigma > 1e10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
