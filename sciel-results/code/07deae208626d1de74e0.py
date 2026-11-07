import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))
        if lambda_ > 2 * n:
            lambda_ = 2 * n
        mu = lambda_ // 2
        weights = np.log((mu + 1) / 2) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (1 + 2 * max(0, np.sqrt((mu_eff / 2 - 1) / (n + 1)) - 1)) / (n + 4)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 1 / mu_eff)) / ((n + 2) ** 2 - mu_eff))
        
        # Initial state
        # Start at 0.0, often a good heuristic for BBOB if shifts are moderate
        # Or random? Let's use random uniform in [-5, 5] to be safe against shifts.
        # Actually, starting at 0 is better if the shift is small. 
        # The problem says "hidden seeded shift... minimizer lies in [-4, 4]^5".
        # If shift is large, 0 is bad. If shift is small, 0 is good.
        # A random start is more robust.
        x_mean = np.zeros(n) # Let's try 0 first, it's symmetric. 
        # Wait, "hidden seeded shift". If the seed is fixed per instance, 
        # maybe 0 is not optimal. 
        # Let's use a random start to avoid bias.
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-5, 5, n)
        
        sigma = 2.0
        C = np.eye(n)
        C_inv_sqrt = np.eye(n) # Inverse square root of C
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        evals = 1
        
        start_time = time.time()
        timeout = 300.0
        safety_margin = 5.0
        
        # Pre-compute Cholesky factor of C
        # We need to update C and its inverse square root every generation.
        # For n=5, this is cheap.
        
        while evals < budget:
            # Check time
            if time.time() - start_time > timeout - safety_margin:
                break
            
            # Generate population
            # We need the square root of C.
            # C is SPD. Use eigendecomposition or Cholesky.
            # For n=5, eigendecomposition is stable and simple.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            Y = []
            for _ in range(lambda_):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (C_sqrt @ z)
                # Clip to box
                y_clipped = np.clip(y, -5, 5)
                Y.append(y_clipped)
            
            # Evaluate
            F = []
            for y in Y:
                if evals >= budget:
                    break
                fy = func(y)
                F.append(fy)
                evals += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if not F:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = np.array(Y[idx[:mu]])
            F_sorted = np.array(F[idx[:mu]])
            
            # Weighted mean
            x_new = np.sum(weights[:, np.newaxis] * Y_sorted, axis=0)
            
            # Update paths
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x_mean) / sigma
            # Need C_inv_sqrt. 
            # C_inv_sqrt = (C_sqrt)^-1
            # Since C_sqrt = V D^0.5 V^T, C_inv_sqrt = V D^-0.5 V^T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_new - x_mean) / sigma)
            
            hsig = 1.0 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals // lambda_)))) < (1 + 2 / (n + 1)) * np.sqrt(np.trace(C)) / np.sqrt(n) else 0.0
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            
            # Update Covariance Matrix
            rank_one = np.outer(x_new - x_mean, x_new - x_mean)
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * (Y_sorted - x_mean)[:, np.newaxis, :] * (Y_sorted - x_mean)[:, np.newaxis, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (hsig - 1 + c_c) * c_c * C) + c_mu * rank_mu
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(np.trace(C)) - 1)) 
            # Correction: standard CMA-ES sigma update is:
            # sigma = sigma * exp(c_sigma * (||ps|| / chi_n - 1))
            # chi_n = sqrt(n) * (1 - 1/(4n) - 1/(25n^2))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) - 1.0 / (25 * n ** 2))
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Regularize C to ensure SPD
            # Symmetrize
            C = (C + C.T) / 2.0
            # Add small identity if needed
            C += 1e-12 * np.eye(n)
            
            x_mean = x_new
            
            # Optional: Restart if stagnation? 
            # With 500 budget, restarts are expensive. 
            # If sigma becomes very small or large, we might be stuck.
            # But CMA-ES is quite robust.
            
        return best_x, best_f
