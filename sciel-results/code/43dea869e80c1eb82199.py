"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        self.deadline = 300.0 # Safety margin from 330s limit

    def __call__(self, func):
        n = self.dim
        # Population size
        lam = 10
        # Number of offspring selected for recombination
        mu = lam // 2
        
        # Weights
        weights = np.log(lam / 2 + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Step size control
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, cs - 1 / (n + 1)) + cs
        
        # Covariance matrix adaptation
        c1 = 2 / ((n + 1.3) ** 2 + lam)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + lam))
        
        # Path length control
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff)
        
        # Initial parameters
        x = np.zeros(n) # Start at center
        sigma = 2.0
        C = np.eye(n)
        Cinv = np.eye(n)
        Csqrt = np.eye(n)
        
        # Precompute sqrt of C for sampling
        # Initially C=I, so Csqrt=I
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        
        # Track elapsed time
        if time.time() - self.start_time > self.deadline:
            return best_x, best_f

        for gen in range(self.budget // lam + 1):
            if calls >= self.budget:
                break
            if time.time() - self.start_time > self.deadline:
                break
            
            # Sample lambda points
            offspring = []
            offspring_f = []
            
            # Update Csqrt if C changed (we do this at the end of prev gen, so it's ready)
            # For efficiency, we can just use np.linalg.cholesky or eigen decomposition
            # Since n=5, eigen is cheap.
            # Actually, let's just use the Cholesky factor if C is positive definite.
            # C is always SPD in CMA-ES.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigen if numerical issues arise
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            
            for i in range(lam):
                if calls >= self.budget:
                    break
                # Sample from N(0, C)
                z = np.random.randn(n)
                y = x + sigma * (L @ z)
                
                # Clip to box [-5, 5]
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                offspring.append(y)
                offspring_f.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) == 0:
                break
                
            # Sort offspring by fitness
            idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in idx]
            offspring_f = [offspring_f[i] for i in idx]
            
            # Select top mu
            y_mu = np.array(offspring[:mu])
            
            # Update mean
            x_old = x.copy()
            x = np.sum(weights[:, None] * y_mu, axis=0)
            
            # Update path length for covariance
            y_diff = (x - x_old) / sigma
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * y_diff
            
            # Update path length for step size
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * mueff) * (Cinv @ y_diff)
            
            # Check if ps is too large
            delta_sig = 1.0 - np.exp(-np.sum(ps ** 2) / n - 1)
            sigma *= np.exp(cs / ds * (np.sum(ps ** 2) / n - 1))
            
            # Update covariance matrix
            # Rank-one update
            v = pc * (1 + cs * (1 - cc)) + (1 - cc) * y_diff
            C = (1 + c1 * delta_sig - c1) * C + c1 * (v @ v.T - C)
            
            # Rank-mu update
            # We need to compute the weighted outer products of the selected points
            # y_mu is mu x n
            # weights is mu
            # C_mu = sum(w_i * (y_i - x_old) @ (y_i - x_old).T)
            # Note: y_i here are the selected offspring, which are already relative to x_old? No, they are absolute.
            # The formula uses (y_i - x_old)
            y_diff_mu = y_mu - x_old
            # Weighted outer products
            C_mu = np.sum(weights[:, None, None] * (y_diff_mu[:, :, None] * y_diff_mu[:, None, :]), axis=0)
            
            C = (1 + c1 * delta_sig + cmu - c1 - cmu) * C + cmu * C_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Cinv and Csqrt
            # For n=5, we can just invert C directly
            try:
                Cinv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                Cinv = np.linalg.pinv(C)
                
            # Update Cholesky factor for next iteration
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                L = eigvecs @ np.diag(np.sqrt(eigvals))

        return best_x, best_f
# EVOLVE-BLOCK-END
