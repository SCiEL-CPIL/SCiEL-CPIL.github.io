"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        nu_eff = 1 / np.sum(weights**2)
        
        # Step size adaptation
        c_sigma = (nu_eff + 2) / (n + nu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((nu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + nu_eff / n) / (n + 4 + 2 * nu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + nu_eff)
        c_mu = min(1 - c_1, 2 * (nu_eff - 2 + 1 / nu_eff) / ((n + 2)**2 + nu_eff))
        
        # Initial step size
        sigma = 2.0
        
        # Initial mean
        rng = np.random.default_rng(42)
        m = rng.uniform(-5, 5, n)
        
        # Covariance matrix
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = m
        best_f = func(m)
        
        # Precompute matrices for speed
        # Update C_sqrt and C_inv_sqrt periodically (every 10 generations to save time)
        
        def update_matrices():
            nonlocal C, C_inv, C_sqrt, C_inv_sqrt
            # Compute square root of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12) # Ensure positive definite
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
        # Initial update
        update_matrices()
        
        calls_made = 1
        gen = 0
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Sample offspring
            ys = []
            fs = []
            
            # We need to sample lam points
            # To save time, we can sample in batches
            # But CMA-ES requires all lam for the update
            
            # Check if we have enough budget for a full generation
            if calls_made + lam > budget:
                # Do a final partial generation or just stop
                # For simplicity, we stop if we can't do a full gen, 
                # but we can do a few more single steps if we want. 
                # However, CMA-ES update requires full pop. 
                # Let's just break if we can't afford a full population.
                break
            
            for i in range(lam):
                z = rng.standard_normal(n)
                y = m + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                calls_made += 1
                
                if time.time() > deadline:
                    # If we run out of time mid-generation, we can't update properly.
                    # We'll just break out.
                    pass
            
            if time.time() > deadline:
                break
                
            ys = np.array(ys)
            fs = np.array(fs)
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0]
            
            # Weighted mean
            m_new = np.dot(weights, ys[:mu])
            
            # Standard deviation
            sigma_old = sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(C_inv_sqrt @ (m_new - m)) / sigma_old - 1))
            
            # Update p_sigma
            hs = (np.linalg.norm(C_inv_sqrt @ (m_new - m)) / (1 - (1 + gen / (5 * (n + lam))) ** (-1)) < 1.4 + 2 / (n + 1))
            p_sigma = (1 - c_sigma) * p_sigma + hs * np.sqrt(c_sigma * (2 - c_sigma) * nu_eff) * C_inv_sqrt @ (m_new - m) / sigma
            
            # Update p_c
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((nu_eff + 1) * (2 - nu_eff)))
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * nu_eff) * (m_new - m) / sigma
            
            # Update C
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = C_inv_sqrt @ (ys[i] - m) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
                
            C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C + c_c * (rank_one - (1 - hsig) * c_c * (2 - c_c) * C) + c_1 * (rank_one + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * rank_mu
            
            # Ensure C is positive definite
            # Simple check: eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < 1e-12):
                # Regularize
                C += 1e-12 * np.eye(n)
            
            m = m_new
            gen += 1
            
            # Update square roots every 10 generations to save time
            if gen % 10 == 0:
                update_matrices()
                
        return best_x, best_f
# EVOLVE-BLOCK-END
