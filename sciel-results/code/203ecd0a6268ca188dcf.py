"""Optimized CMA-ES for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(d))
        if lam < 4: lam = 4
        # Number of parents
        mu = lam // 2
        
        # Weights
        # Standard weights: w_i = log(mu + 0.5) - log(i)
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 0.5) - np.log(i + 1)
        w = w / w.sum()
        
        # Variance equivalent population size
        mueff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        cs = (mueff + 2) / (d + mueff + 5)
        cc = (4 - mueff + cs) / (d + 5 - mueff + cs)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(c1, 2 * (mueff - 2 + 1/mueff) / ((d + 2)**2 + mueff))
        
        # Initialization
        # Start at center, large sigma to explore [-5, 5]^d
        xmean = np.zeros(d)
        sigma = 4.0 # Initial step size
        
        # State variables
        C = np.eye(d)
        Cinv_sqrt = np.eye(d) # Inverse square root of C
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Pre-compute Cholesky factorization update helper
        # We will update C and recompute Cinv_sqrt every generation or use a more efficient method.
        # For d=5, recomputing is cheap.
        
        def chol_update(C, Cinv_sqrt, pc, ps, w, mu, cs, cc, c1, cmu, d):
            # This is a simplified update. In a full implementation, one would update C incrementally.
            # For d=5, we can just update C and recompute.
            # However, the standard CMA-ES update formula for C is:
            # C <- (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - (1 - cs)^2) * C) + cmu * sum(wi * yi * yi.T)
            # where yi are the selected steps.
            
            # We need the selected steps (yi) to compute the cmu part.
            # This function is a placeholder for the logic inside the loop.
            pass

        # We will manage the loop manually to keep it simple and efficient for d=5.
        
        best_x = xmean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # If budget is too small for even one generation, we just return
        if budget < lam:
             # Do a simple random search or gradient descent if possible? 
             # With < lam calls, we can't do a proper CMA-ES generation.
             # Just do random steps.
             while calls_made < budget:
                 step = np.random.randn(d) * sigma
                 new_x = np.clip(xmean + step, -5, 5)
                 f_new = func(new_x)
                 calls_made += 1
                 if f_new < best_f:
                     best_f = f_new
                     best_x = new_x
             return best_x, best_f

        # Main CMA-ES Loop
        start_time = time.time()
        
        while calls_made < budget:
            # 1. Sample lambda offspring
            # Update Cinv_sqrt if C has changed (it changes every generation)
            # To save time, we can update Cinv_sqrt less frequently, but for d=5 it's fast.
            
            # Generate random normal vectors
            # z_i ~ N(0, I)
            Z = np.random.randn(lam, d)
            
            # Transform to y_i = B * D * z_i. Here B*D is the square root of C.
            # We maintain Cinv_sqrt, so we can get C_sqrt? No, we need C_sqrt to sample.
            # Standard CMA-ES maintains C and computes its eigenvalue decomposition or Cholesky.
            # For simplicity and robustness in d=5, we will compute Cholesky of C every generation.
            # C is symmetric positive definite.
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset it
                C = np.eye(d)
                C_chol = np.eye(d)
            
            # Sample Y = Z @ C_chol.T
            Y = Z @ C_chol.T
            
            # Offspring X
            X = np.tile(xmean, (lam, 1)) + sigma * Y
            # Clip to bounds [-5, 5]
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lam
            
            # Check time
            if time.time() - start_time > 300:
                break
                
            # 2. Select parents
            # Sort indices by fitness
            idx = np.argsort(F)
            I = idx[:mu]
            
            # Update best
            if F[idx[0]] < best_f:
                best_f = F[idx[0]]
                best_x = X[idx[0]].copy()
                
            # 3. Update mean
            # Weighted sum of parents
            # xmean_new = sum(w_i * x_{i:lambda})
            xmean_new = np.sum(w[:, None] * X[I], axis=0)
            
            # 4. Update step size sigma
            # ps = (1 - cs) * ps + cs * sqrt(mueff) * C^{-1/2} * (xmean_new - xmean) / sigma
            # We need C^{-1/2}. We have C_chol. C^{-1/2} is not directly C_chol^{-1}.
            # Actually, if C = L L^T, then C^{-1/2} is not simply L^{-1}.
            # However, the standard update uses the "natural gradient" which involves C^{-1/2}.
            # Let's compute C_inv_sqrt properly.
            # C_inv_sqrt = (C^{-1/2})
            # We can compute eigenvalues/vectors or use Cholesky.
            # C = L L^T. C^{-1} = L^{-T} L^{-1}.
            # C^{-1/2} is the matrix square root of C^{-1}.
            # For d=5, we can just compute it via eigendecomposition of C.
            # Or, simpler: use the fact that C_inv_sqrt @ C_inv_sqrt = C_inv.
            
            # Let's compute C_inv and C_inv_sqrt
            C_inv = np.linalg.inv(C)
            # Symmetric square root of inverse
            # Use eigendecomposition
            w_eig, v_eig = np.linalg.eigh(C_inv)
            # Clip negative eigenvalues due to numerical errors
            w_eig = np.clip(w_eig, 1e-12, None)
            C_inv_sqrt = v_eig @ np.diag(np.sqrt(w_eig)) @ v_eig.T
            
            # Update ps
            ps = (1 - cs) * ps + cs * np.sqrt(mueff) * (C_inv_sqrt @ (xmean_new - xmean)) / sigma
            
            # Update sigma
            # chi_n = E[||N(0, I)||] = sqrt(d) * (1 - 1/(4d) + 1/(32d^2))
            chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(32*d**2))
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # 5. Update covariance matrix C
            # pc = (1 - cc) * pc + cc * sqrt(mueff) * (xmean_new - xmean) / sigma
            pc = (1 - cc) * pc + cc * np.sqrt(mueff) * (xmean_new - xmean) / sigma
            
            # C update
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - (1 - cs)^2) * C) + cmu * sum(wi * yi * yi.T)
            # yi = (x_i - xmean_old) / sigma
            Y_parents = (X[I] - xmean) / sigma
            
            # Compute the weighted sum of outer products
            # sum_i w_i * y_i * y_i.T
            # This is efficient for small mu
            C_mu_part = np.zeros((d, d))
            for i in range(mu):
                C_mu_part += w[i] * np.outer(Y_parents[i], Y_parents[i])
                
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - cs)**2) * C) + cmu * C_mu_part
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            xmean = xmean_new
            
            # If we hit the budget exactly or exceeded, break
            if calls_made >= budget:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
