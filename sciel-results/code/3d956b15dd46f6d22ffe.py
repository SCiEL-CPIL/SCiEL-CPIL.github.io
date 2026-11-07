import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For d=5, lambda=10 is a good balance
        lam = 10
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.arange(mu)
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        # Strategy parameters
        cs = (mueff + 2) / (d + mueff + 5)
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c1 = 2 / (d**2 + 18)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / (d**2 + 18))
        
        # Initial state
        # Start near center, but maybe slightly offset to break symmetry if needed? 
        # Standard start is 0.
        x_mean = np.zeros(d)
        sigma = 1.0  # Initial step size
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Precompute inverse square root of C
        # We need to update this every generation
        def inv_sqrt(C_mat):
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            inv_sqrt_eig = 1.0 / np.sqrt(eigvals)
            return eigvecs @ np.diag(inv_sqrt_eig) @ eigvecs.T

        B_inv_sqrt = inv_sqrt(C)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Main Loop
        # We have budget calls left
        # Each generation uses lam calls
        # Total generations = budget // lam
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            # y_i ~ N(0, I)
            Y = np.random.standard_normal((lam, d))
            # Transform to x_i
            X = x_mean + sigma * (Y @ B_inv_sqrt.T)
            
            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                x_i = X[i]
                # Clip to box
                x_i = np.clip(x_i, -5, 5)
                F[i] = func(x_i)
                calls += 1
            
            # If we ran out of budget mid-generation, break
            if calls >= budget:
                break
                
            # Sort by function values
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                
            # Update mean
            x_mean_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update evolution path for covariance (pc)
            # Need B and D (from C = B B^T)
            # For efficiency in small d, we can just use the eigendecomposition again or update C directly
            # Standard CMA-ES updates C using B and D. 
            # Let's stick to the standard update rule which requires B and D.
            # C = B diag(D^2) B^T
            eigvals, B_mat = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            D = np.sqrt(eigvals)
            
            # ps update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B_mat @ (D * (B_mat.T @ (x_mean_new - x_mean) / sigma)))
            
            # pc update
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls // lam))) < (1.4 + 2 / (d + 1)) * np.sqrt(1 - (1 - cs)**(2 * (calls // lam))))
            
            pc = (1 - cc) * pc + hsig * cc * (x_mean_new - x_mean) / sigma
            
            # Update C
            delta_hs = hsig * (1 - cc) - (1 - hsig) * cc
            C = (1 + c1 * delta_hs - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            C += cmu * np.sum(w[:mu, np.newaxis, np.newaxis] * np.outer(X_sorted[:mu] - x_mean, X_sorted[:mu] - x_mean) / sigma**2, axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls // lam))) - chi_d))
            # chi_d is the expected norm of standard normal
            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
            d_sigma = 1 + 2 * max(0, (mueff - d - 1) / (d + mueff - 3))
            
            # Check if sigma is too large or small
            if sigma > 100:
                sigma = 100
            if sigma < 1e-10:
                break
                
            x_mean = x_mean_new
            
            # Update B_inv_sqrt for next iteration
            # We need to update B_inv_sqrt = (B diag(D) B^T)^(-1/2)
            # This is computationally expensive if done every iteration, but d=5 is small.
            B_inv_sqrt = inv_sqrt(C)
            
        return best_x, best_f
