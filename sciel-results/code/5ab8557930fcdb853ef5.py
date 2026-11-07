"""Rewritten optimizer for BBOB f13 (Sharp ridge) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        w_mu = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w_mu = w_mu / w_mu.sum()
        
        mu_eff = 1.0 / np.sum(w_mu**2)
        
        cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c1 = 2.0 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + np.sqrt(2.0)) / ((n + 2.0)**2 + mu_eff))
        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + cs
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))
        
        calls_made = 0
        best_fx = float('inf')
        best_x = np.zeros(n)
        
        def eval_point(x):
            nonlocal calls_made, best_fx, best_x
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx
        
        eval_point(x_mean)
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            ys = np.zeros((lam, n))
            fs = np.zeros(lam)
            
            # Calculate Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
            
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline:
                    break
                z = np.random.randn(n)
                y = x_mean + sigma * (L @ z)
                ys[i] = y
                fs[i] = eval_point(y)
            
            # Sort
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]
            
            # Update mean
            x_mean_new = np.sum(w_mu[:, np.newaxis] * ys[:mu], axis=0)
            
            # Update step size
            # pc = (1 - cs) * pc + sqrt(cs * (2 - cs) * mu_eff) * (L @ (x_mean_new - x_mean) / sigma)
            # Note: The standard CMA-ES uses the untransformed difference for pc? 
            # Actually, pc is in the "search space" defined by C.
            # Standard: pc = (1-cs)pc + sqrt(cs(2-cs)mu_eff) * (1/sigma) * (x_new - x_old)
            # But often implemented with the transformation to keep it in the unit sphere space?
            # Hansen's original: pc is updated with the difference of means, scaled by 1/sigma.
            # ps is updated with the Cholesky transformed difference.
            
            # Let's stick to the standard implementation from Hansen's paper/code.
            # pc update:
            pc = (1.0 - cs) * pc + np.sqrt(cs * (2.0 - cs) * mu_eff) * (x_mean_new - x_mean) / sigma
            
            # ps update:
            # ps = (1-cs)ps + sqrt(cs(2-cs)mu_eff) * (L^{-1} @ (x_mean_new - x_mean) / sigma) ?
            # No, L is sqrt(C). The step is sigma * L * z.
            # The direction in the "z-space" is z.
            # The mean shift in z-space is L^{-1} @ (x_mean_new - x_mean) / sigma.
            # Let's compute L_inv.
            try:
                L_inv = np.linalg.inv(L)
            except:
                L_inv = np.eye(n)
                
            z_mean_shift = L_inv @ (x_mean_new - x_mean) / sigma
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mu_eff) * z_mean_shift
            
            # Check for step size increase
            norm_ps = np.linalg.norm(ps)
            # The condition uses the expected norm
            # t_sigma is the age of the strategy, approximated by calls_made / (lam * (1-cs)) ? 
            # Simpler: just use the norm check
            if norm_ps / np.sqrt(1.0 - (1.0 - cs)**(2.0 * max(1, int(calls_made / lam)))) < (1.4 + 2.0 / (n + 1.0)) * chi_n:
                sigma *= np.exp(cs / ds * (norm_ps / chi_n - 1.0))
            
            # Update covariance
            # Rank-1 update
            h_sig = 1.0 if norm_ps / np.sqrt(1.0 - (1.0 - cs)**(2.0 * max(1, int(calls_made / lam)))) < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0
            C = (1.0 - c1 - cmu) * C + c1 * (np.outer(ps, ps) + h_sig * (norm_ps**2 - np.trace(C)) * np.eye(n))
            
            # Rank-mu update
            v = (ys[:mu] - x_mean) / sigma
            # v is in the original space? No, we need the update in the C-space.
            # The standard update adds c_mu * sum w_i (y_i - x_mean)(y_i - x_mean)^T / sigma^2
            # This is in the original coordinate system.
            C += cmu * np.sum(w_mu[:, np.newaxis, np.newaxis] * np.outer(v, v), axis=0)
            
            # Symmetrize
            C = 0.5 * (C + C.T)
            
            x_mean = x_mean_new
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
