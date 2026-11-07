"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        
        # Initialize
        x = rng.uniform(-5, 5, self.dim)
        sigma = 2.0
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        # CMA-ES parameters
        mu = self.dim // 2 + 1
        lambda_pop = 2 * mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        
        budget_used = 1
        best_x = x
        best_f = func(x)
        
        # Pre-compute cholesky inverse for C
        try:
            B = np.linalg.cholesky(C)
            B_inv = np.linalg.inv(B)
        except:
            B = np.eye(self.dim)
            B_inv = np.eye(self.dim)

        for gen in range(self.budget // lambda_pop + 1):
            if time.time() > deadline:
                break
                
            # Generate population
            ys = []
            fvals = []
            for i in range(lambda_pop):
                # z ~ N(0, I)
                z = rng.standard_normal(self.dim)
                # y = x + sigma * B * z
                y = x + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                budget_used += 1
                if budget_used > self.budget:
                    break
                ys.append(y)
                fvals.append(fy)
                
            if budget_used > self.budget:
                break
                
            # Sort
            idx = np.argsort(fvals)
            ys = [ys[i] for i in idx]
            fvals = [fvals[i] for i in idx]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = ys[0]
                
            # Compute new center
            x_new = np.sum(np.array(weights)[:, np.newaxis] * np.array(ys[:mu]), axis=0)
            
            # Update covariance matrix
            y_inv = np.array(ys[:mu]) - x
            # B_inv @ y_inv
            y_inv_B_inv = (B_inv.T @ y_inv.T).T
            
            # Update C
            C += c_1 * (np.outer(pc, pc) - C) + c_mu * (y_inv_B_inv.T @ y_inv_B_inv / mu_eff)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update pc
            pc = (1 - c_sigma) * pc + c_sigma * np.sqrt(mu_eff) * (B_inv @ (x_new - x))
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B_inv @ (x_new - x))
            
            # Update sigma
            sigma *= np.exp(d_sigma / c_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Update x
            x = x_new
            
            # Update B and B_inv
            try:
                B = np.linalg.cholesky(C)
                B_inv = np.linalg.inv(B)
            except:
                # Regularize if not positive definite
                C += 1e-12 * np.eye(self.dim)
                try:
                    B = np.linalg.cholesky(C)
                    B_inv = np.linalg.inv(B)
                except:
                    B = np.eye(self.dim)
                    B_inv = np.eye(self.dim)
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
