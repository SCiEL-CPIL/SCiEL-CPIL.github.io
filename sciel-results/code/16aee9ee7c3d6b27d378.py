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
        deadline = start_time + 290.0
        
        # (1+1)-CMA-ES Parameters
        # c_sigma, c_c, c_cov, d_sigma are standard for (1+1)-ES
        c_sigma = (1 + 2 / (dim + 1)) / (dim + 4 + 2 / (dim + 1))
        c_c = (4 + 1 / dim) / (dim + 4 + 2 / (dim + 1))
        c_cov = (2 + 2 / dim) / (dim + 1 + 2 / dim)
        d_sigma = 1 + 2 * max(0, (np.sqrt(dim) - 1) / (dim + 3) - 1) + c_sigma
        
        # State
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        rng = np.random.default_rng(42)
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sample
            z = rng.standard_normal(dim)
            try:
                L = np.linalg.cholesky(C)
            except:
                C = np.eye(dim)
                L = np.eye(dim)
            
            y = sigma * (L @ z)
            x_new = np.clip(x + y, -5, 5)
            fx_new = func(x_new)
            calls += 1
            
            # Accept/Reject
            if fx_new < fx:
                # Update mean
                x_old = x.copy()
                x = x_new
                fx = fx_new
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Update paths
                # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu) * (x - x_old) / (sigma * ||C^{1/2}||_F)
                # mu = 1
                c_norm = np.sqrt(np.trace(C))
                if c_norm == 0: c_norm = 1.0
                
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (x - x_old) / (sigma * c_norm)
                
                # sigma adaptation
                e_norm = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / e_norm - 1))
                
                # p_c update
                p_c = (1 - c_c) * p_c + c_c * (x - x_old) / sigma
                
                # C update
                # C = (1 - c_cov * (1 + (1 - nu) * c_sigma * (1 - c_sigma))) * C + c_cov * (nu * p_sigma * p_sigma.T + y * y.T)
                # For (1+1), nu = 1, w=1
                # rank_one = y * y.T where y = (x - x_old) / sigma
                y_vec = (x - x_old) / sigma
                rank_one = np.outer(y_vec, y_vec)
                
                c_cov_eff = c_cov * (1 + c_sigma * (1 - c_sigma)) # nu=1
                C = (1 - c_cov_eff) * C + c_cov_eff * (np.outer(p_sigma, p_sigma) + rank_one)
                C = (C + C.T) / 2
            else:
                # Reject
                # Update paths even on rejection? 
                # In standard (1+1)-ES, paths are updated only on acceptance?
                # No, CMA-ES updates paths based on the step, regardless of acceptance?
                # Actually, in (1+1)-ES, if rejected, the step is not taken, so x doesn't change.
                # But the "attempted" step y is known.
                # Standard (1+1)-ES:
                # If rejected, x stays same.
                # p_sigma and p_c are updated using the attempted step?
                # Literature says: "The paths are updated with the step y if accepted, and with -y if rejected? No."
                # Usually, in (1+1)-ES, the update of C and sigma happens only on acceptance?
                # Or is it every step?
                # Let's check standard (1+1)-ES pseudocode.
                # Usually, adaptation parameters are updated every generation, using the step y (which is 0 if rejected? No, y is the step taken).
                # If rejected, the step taken is 0. So y_vec = 0.
                # Then p_sigma update uses 0? That would decay p_sigma.
                # This is a valid strategy.
                
                y_vec = np.zeros(dim)
                p_sigma = (1 - c_sigma) * p_sigma
                p_c = (1 - c_c) * p_c
                # C update with y_vec=0
                c_cov_eff = c_cov * (1 + c_sigma * (1 - c_sigma))
                C = (1 - c_cov_eff) * C + c_cov_eff * (np.outer(p_sigma, p_sigma))
                C = (C + C.T) / 2
                
                # Sigma adaptation still happens?
                # Yes, based on p_sigma.
                e_norm = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / e_norm - 1))
            
            # Clip x to bounds (already done in x_new)
            # Ensure C is positive definite
            # Add small regularization if needed
            
        return best_x, best_f
