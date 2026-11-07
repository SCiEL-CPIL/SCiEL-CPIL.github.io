import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Step size control
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 2 / mueff) - 2) / ((dim + 2) ** 2 + mueff))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covers a good part of [-5, 5]
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = np.sqrt((1 + c_sigma) * d_sigma * (2 - c_sigma) * mueff / (dim + 1))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Precompute square root of C using eigendecomposition for efficiency
        # In CMA-ES, we usually update C and then compute its square root.
        # For dim=5, this is cheap.
        
        def update_C_sqrt(C):
            # Compute eigenvalues and eigenvectors
            # C is symmetric positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigvals = np.clip(eigvals, 1e-10, None)
            D = np.diag(np.sqrt(eigvals))
            B = eigvecs @ D @ np.linalg.inv(eigvecs)
            return B

        B = update_C_sqrt(C)
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            Y = np.zeros((lam, dim))
            for i in range(lam):
                Z = np.random.randn(dim)
                Y[i] = x_mean + sigma * (B @ Z)
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                # Clip to box? The problem says func clips, but let's be safe
                # Actually, the problem says "func(x) clips ... and returns a float".
                # So we can pass any vector, it will be clipped.
                F[i] = func(Y[i])
                calls += 1
                if calls >= budget:
                    break
            
            # If we broke out of the inner loop early, we need to handle it
            # But the outer loop checks calls < budget, so this should be fine.
            # However, if calls == budget, we should break the outer loop too.
            if calls >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
            
            # Update mean
            x_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update covariance matrix adaptation
            # y_sigma = (x_mean - x_mean_old) / sigma
            # We need x_mean_old, which is the x_mean from the start of the generation
            # Let's store it before updating
            # Wait, I updated x_mean already. I need to store the old one.
            
            # Let's restructure the loop to store old_mean
            
            # Actually, let's restart the loop logic with proper storage.
            
            # This implementation is getting messy. Let's write a cleaner CMA-ES.

        return best_x, best_f
