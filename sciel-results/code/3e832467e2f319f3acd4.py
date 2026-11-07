"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: kill at 280s, limit is 300s (evaluator) / 330s (hard kill)
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # lambda: population size
        lam = 4 + int(3 * np.log(dim))
        # mu: number of best solutions used for recombination
        mu = lam // 2
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 / ((dim + 1.3)**2 + mueff)) * (2 - cc)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)))
        
        # Initial state
        # Center of the box is a reasonable start
        x_mean = np.zeros(dim)
        
        # Initial sigma based on box size
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C
        C_inv_sqrt = np.eye(dim)
        
        # State variables for adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Step size control
        d_sigma = 1 + 0.3 * min(mueff / dim, 2) + cs
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Evaluation counter
        count = 0
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # First evaluation
        try:
            best_f = func(best_x)
            count += 1
        except Exception:
            # If even first call fails, return the initial guess to avoid total failure
            # The contract says exception before first call invalidates, but returning 
            # a valid result is better than crashing if possible. 
            # However, if func is broken, we can't do much.
            pass

        # Main loop
        while count < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for i in range(lam):
                # Standard normal
                z = np.random.randn(dim)
                # Transform
                y = x_mean + sigma * (C_inv_sqrt @ z)
                # Clip to box [-5, 5]
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            # Evaluate population
            fs = []
            eval_failed = False
            for y in ys:
                try:
                    f = func(y)
                    count += 1
                    if f < best_f:
                        best_f = f
                        best_x = y.copy()
                    fs.append(f)
                except Exception:
                    eval_failed = True
                    break
            
            if count >= budget:
                break
                
            # Check if all evaluations succeeded
            if eval_failed or len(fs) < lam:
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = np.array(ys)[order]
            fs_sorted = np.array(fs)[order]
            
            # Update mean
            x_mean_new = sum(w[i] * ys_sorted[i] for i in range(mu))
            
            # Calculate y_p (step in normalized space)
            y_p = (x_mean_new - x_mean) / sigma
            
            # Update C (Covariance Matrix)
            # We need to update C and its inverse square root
            # This is computationally heavy for dim=5, but 5 is small.
            
            # Compute p_sigma and p_c
            # p_sigma = (1-cs) * p_sigma + sqrt(cs*(2-cs)*mueff) / sigma * y_p
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_p
            
            # p_c = (1-cc) * p_c + sqrt(cc*(2-cc)*mueff) * y_p
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * y_p
            
            # Update sigma
            sigma *= np.exp(cs / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            # C = (1 - c1 - cmu) * C + c1 * (p_c * p_c.T + (1 - h_sigma) * C) + cmu * sum(w_i * (y_i - y_mean) * (y_i - y_mean).T)
            
            # h_sigma is a heuristic for the norm of p_sigma
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**((count // lam + 1) * 2 * cs)) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            # Calculate the weighted outer products for the selected points
            # y_i are the raw steps (before clipping, but we clipped them). 
            # CMA-ES usually assumes unbounded. Clipping distorts the distribution.
            # For dim=5 and box [-5,5], if sigma is small, clipping is rare.
            # If sigma is large, clipping is frequent and CMA-ES adapts poorly.
            # However, this is a standard approach for bounded CMA-ES approximations.
            
            # y_i - x_mean (in raw space)
            # We need the vectors d_i = (y_i - x_mean) / sigma
            # But we clipped y_i. Let's use the clipped y_i for the update, 
            # acknowledging the distortion.
            
            diff = np.zeros((dim, dim))
            for i in range(mu):
                d = (ys_sorted[i] - x_mean) / sigma
                diff += w[i] * np.outer(d, d)
                
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * C) + cmu * diff
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update C_inv_sqrt
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            x_mean = x_mean_new
            
        return best_x, best_f
