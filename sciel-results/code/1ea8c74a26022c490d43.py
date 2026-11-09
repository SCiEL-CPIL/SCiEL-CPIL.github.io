"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Robust CMA-ES optimizer for BBOB f11 (Discus).
    Uses a large initial population for fast global progress (AOCC) 
    and covariance adaptation for precise local convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s/330s limit
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Initial sigma: The box is [-5, 5]^5. Min is in [-4, 4]^5.
        # A sigma of 2.0 covers the box well initially.
        sigma = 2.0
        
        # Population size: Default is 4 + floor(3 ln n). For n=5, 4+3*1.6=8.
        # For AOCC (early progress), a larger population is better.
        # Let's use lambda = 16 for a good balance of exploration and exploitation.
        # With budget 500, we can afford ~30 generations.
        lam = 16
        
        # Weights
        mu = lam // 2  # Number of parents
        # Logarithmic weights
        w = np.log(np.arange(mu, 0, -1) + 0.5) - np.log(np.arange(mu, 0, -1) + 0.5)
        w = np.log(np.arange(mu, 0, -1) + 0.5)
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # CMA-ES parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # State
        # Start at center (0,0,0,0,0) as it's the most likely region for shifted minima in BBOB
        mean = np.zeros(dim)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = mean.copy()
        best_f = float('inf')
        
        def call_f(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            return func(x)

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Helper to compute inverse square root of matrix
        def compute_c_inv_sqrt(C_mat):
            # Use SVD for stability
            U, s, Vt = np.linalg.svd(C_mat)
            s_inv_sqrt = 1.0 / np.sqrt(s + 1e-12)
            return U @ np.diag(s_inv_sqrt) @ Vt

        # Initial evaluation of the mean (center)
        f_mean = call_f(mean)
        if f_mean is None:
            return best_x, best_f
        update_best(mean, f_mean)
        
        # Main CMA-ES Loop
        while n_calls < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                # Check if we have enough budget for this full generation
                if n_calls + lam > budget:
                    # Adjust lambda for the last partial generation
                    break
                
                # Sample from N(mean, sigma^2 * C)
                # z ~ N(0, I)
                z = rng.standard_normal(dim)
                y = mean + sigma * (C_inv_sqrt @ z)
                
                # Clip to box [-5, 5]
                y = np.clip(y, -5.0, 5.0)
                
                fy = call_f(y)
                if fy is None:
                    break
                
                offspring.append(y)
                offspring_f.append(fy)
            
            if not offspring:
                break
            
            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)
            
            # Sort by fitness
            idx = np.argsort(offspring_f)
            offspring = offspring[idx]
            offspring_f = offspring_f[idx]
            
            # Update best
            if offspring_f[0] < best_f:
                update_best(offspring[0], offspring_f[0])
            
            # Select parents
            mu_used = min(mu, len(offspring))
            parents = offspring[:mu_used]
            
            # Weights for selected parents (renormalize if mu_used < mu)
            if mu_used < mu:
                w_used = w[:mu_used]
                w_used = w_used / w_used.sum()
                mu_eff_used = 1.0 / np.sum(w_used**2)
            else:
                w_used = w
                mu_eff_used = mu_eff
            
            # Update mean
            mean_new = np.sum(w_used[:, np.newaxis] * parents, axis=0)
            
            # Update path evolution strategies
            # pc = (1 - cc) * pc + cc * sqrt(mu_eff) * C_inv_sqrt * (mean_new - mean) / sigma
            pc = (1 - cc) * pc + cc * np.sqrt(mu_eff_used) * (C_inv_sqrt @ (mean_new - mean) / sigma)
            
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt * (mean_new - mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff_used) * (C_inv_sqrt @ (mean_new - mean) / sigma)
            
            # Update covariance matrix
            # delta_hs = (1 + (2 - theta_hs) * cs) * (1 - cs)
            # theta_hs = 1.0 (standard)
            delta_hs = (1 + (2 - 1.0) * cs) * (1 - cs)
            hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (n_calls // lam + 1))) < (1.4 + 2 / (dim + 1)) * ccs) if False else False
            # Simplified h_sigma check (standard CMA-ES)
            ccs = cs
            # Check for stagnation in path
            norm_ps = np.linalg.norm(ps)
            # Approximation for h_sigma
            h_sigma = 1.0 if norm_ps < (1.4 + 2.0 / (dim + 1.0)) * ccs else 0.0
            
            # Rank-one update
            yk = (mean_new - mean) / sigma
            C = (1 + c1 * h_sigma * cc * (cc - 1) - c1 * cmu) * C + c1 * np.outer(yk, yk)
            
            # Rank-mu update
            # yk_i = (parents[i] - mean) / sigma
            yk_parents = (parents - mean) / sigma
            C = C + cmu * np.sum(w_used[:, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->ijk', w_used, yk_parents, yk_parents), axis=0)
            
            # Ensure symmetry and positive definite (regularize if needed)
            C = (C + C.T) / 2.0
            # Simple regularization: add small identity if eigenvalues are small
            # Check condition number roughly
            # For robustness in short runs, we can skip complex PD checks if we regularize C_inv_sqrt
            C_inv_sqrt = compute_c_inv_sqrt(C)
            
            mean = mean_new
            
            # Adapt sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
            # Limit sigma to avoid explosion or vanishing
            sigma = np.clip(sigma, 1e-8, 50.0)
            
            # Stagnation check for restart (simplified iPOP)
            # If best_f hasn't improved in X generations, or sigma is too small
            # For BBOB f11, a single run is often sufficient if started well.
            # However, if we detect convergence (sigma < 1e-4), we might restart.
            # Given budget 500 and lam=16, we have ~30 gens. 
            # If sigma drops below 0.01, we are likely stuck.
            if sigma < 0.01:
                # Simple restart: perturb mean slightly and reset sigma
                # But we must be careful not to waste calls.
                # Let's just break and do a final local search if needed, 
                # or just continue with tiny sigma.
                # Actually, breaking and doing a (1+1)-ES polish is safer.
                break

        # Final polish with (1+1)-ES if we have budget left and time
        remaining = budget - n_calls
        if remaining > 10 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma_poly = 0.1
            for _ in range(remaining):
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma_poly * rng.standard_normal(dim), -5.0, 5.0)
                fy = call_f(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma_poly *= 1.1
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx
                else:
                    sigma_poly *= 0.9
                    sigma_poly = max(sigma_poly, 1e-4)

        return best_x, best_f

# EVOLVE-BLOCK-END
