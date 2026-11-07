import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = int(4 + 3 * np.log(n))
        if lambda_ > budget:
            lambda_ = budget
        mu = max(1, lambda_ // 2)
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
        
        # Initialization
        x = np.zeros(n)
        sigma = 2.0
        B = np.eye(n)
        D = np.ones(n) # Initial D is 1, so C = I
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x.copy()
        best_f = func(x)
        
        calls = 1
        
        # Main Loop
        while calls + lambda_ <= budget:
            ys = []
            fs = []
            
            # Sampling
            for i in range(lambda_):
                z = np.random.randn(n)
                y = x + sigma * (B @ (D * z))
                # Optional: Clip to box to avoid wasting calls on far-out points?
                # The function clips anyway. Clipping here ensures we stay in the search space.
                # y = np.clip(y, -5, 5) 
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                calls += 1
                
            ys = np.array(ys)
            fs = np.array(fs)
            
            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Weighted mean
            m_new = np.sum(weights[:, None] * ys[:mu], axis=0)
            
            # Cumulation step
            # Step direction in the original space
            step = m_new - x
            
            # Transform step to C-space (whitened space)
            # y = x + sigma * B D z => z = (1/sigma) D^-1 B^-1 (y-x)
            # Since B is orthogonal, B^-1 = B.T
            # z_step = (1/sigma) * np.diag(1/D) @ B.T @ step
            z_step = (1.0 / sigma) * (B.T @ step) / D
            
            # Update pc and ps
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * z_step
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_step
            
            # Covariance Matrix Update
            # Current C
            C = (B * D) @ B.T
            
            # Delta C
            delta = (1 - c_1 - c_mu) * C
            delta += c_1 * (np.outer(pc, pc) - C)
            
            # Rank-mu update
            # sum(weights_i * (y_i - m) (y_i - m)^T)
            # We need the whitened differences for the rank-mu update?
            # Standard CMA-ES updates C in the original space.
            # diff = ys[:mu] - m_new
            # rank_mu_term = sum(weights_i * outer(diff_i, diff_i))
            # delta += c_mu * rank_mu_term
            
            # However, the pc and ps updates above were done in z-space.
            # The standard algorithm updates C using the z-space vectors for the rank-1 part?
            # No, the standard formula is:
            # C <- (1 - c_1 - c_mu) C + c_1 (pc pc^T - C) + c_mu sum w_i (y_i - m)(y_i - m)^T
            # Here pc and ps are in the "z" space?
            # In the standard implementation (Hansen), pc and ps are in the original space?
            # Let's check.
            # y_i = x + sigma B D z_i
            # m = x + sigma B D z_bar
            # z_bar = sum w_i z_i
            # step = m - x = sigma B D z_bar
            # pc is updated with z_bar?
            # Hansen's paper: pc <- (1-c_c)pc + c_c/sigma * (B D (m - x)) ??
            # Actually, pc is maintained in the "z" space (standard normal space) in some implementations,
            # or in the original space in others.
            # If pc is in z-space:
            # pc_new = (1-c_c)pc_old + sqrt(c_c(2-c_c)mu_eff) * z_bar
            # Then the rank-1 update to C is c_1 * (B D pc_new (B D pc_new)^T - C)?
            # Or c_1 * (pc_new pc_new^T - I) in z-space?
            
            # Let's stick to the most common implementation:
            # pc and ps are in the "z" space (whitened space).
            # C is in the original space.
            # The update to C is:
            # C_new = (1 - c_1 - c_mu) C + c_1 * (B D pc pc^T D B^T - C) + c_mu * sum(w_i (y_i - m)(y_i - m)^T)
            
            # Let's re-calculate delta in original space.
            # Rank-1 term in original space:
            # v = B @ (D * pc)
            # rank1 = np.outer(v, v)
            
            # Rank-mu term in original space:
            # diff = ys[:mu] - m_new
            # rank_mu = np.sum(weights[:, None] * diff[:, :, None] * diff[:, None, :], axis=0)
            
            # Re-initialize delta
            delta = (1 - c_1 - c_mu) * C
            v = B @ (D * pc)
            delta += c_1 * (np.outer(v, v) - C)
            diff = ys[:mu] - m_new
            rank_mu = np.einsum('i,ij,ik->jk', weights, diff, diff)
            delta += c_mu * rank_mu
            
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(delta)
            eigvals = np.clip(eigvals, 1e-10, None)
            
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # Update x and sigma
            x = m_new
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Check for convergence or issues
            if sigma < 1e-10 or not np.all(np.isfinite(x)):
                break

        return best_x, best_f
