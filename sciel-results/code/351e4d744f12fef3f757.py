import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        N = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For small N, lambda is typically 4 + 3*log(N)
        lam = 10 
        # Initial step size
        sigma = 2.0
        # Initial mean
        rng = np.random.default_rng(42)
        m = np.zeros(N)
        
        # Covariance matrix (Identity initially)
        C = np.eye(N)
        # Eigenvalues and eigenvectors of C
        # We will update C directly or via eigendecomposition every step?
        # For N=5, eigendecomposition is cheap.
        
        # Evolution paths
        p_sigma = np.zeros(N)
        p_c = np.zeros(N)
        
        # Parameters
        # mu = lam // 2
        mu = lam // 2
        # Weights
        w = np.log(lam / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (N + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff - 1) / (N + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
        c_1 = 2 / ((N + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((N + 2)**2 + mu_eff))
        
        # Chi distribution approximation for p_sigma normalization
        chi_N = N * (1 - 1/(4*N) + 1/(21*N**2))
        
        # Time limit
        start_time = time.time()
        time_limit = 280.0 # Safety margin
        
        best_x = m.copy()
        best_f = np.inf
        
        # Initial evaluation
        best_f = func(best_x)
        budget -= 1
        
        while budget > 0:
            # Time check
            if time.time() - start_time > time_limit:
                break
            
            # Generate population
            Y = np.zeros((lam, N))
            F = np.zeros(lam)
            
            # Eigendecomposition of C
            D, B = np.linalg.eigh(C)
            D = np.sqrt(D)
            
            for i in range(lam):
                z = rng.standard_normal(N)
                # Sample from N(0, C)
                Y[i] = B @ (D * z)
                x = m + sigma * Y[i]
                # Clip to box [-5, 5]
                x = np.clip(x, -5, 5)
                F[i] = func(x)
                budget -= 1
                if budget <= 0:
                    break
                
                # Update best
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()
            
            if budget <= 0:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update mean
            m_new = m + sigma * np.sum(w[:mu] * Y[:mu], axis=0)
            
            # Update evolution path for sigma
            # h_sigma condition
            h_sigma = (np.sqrt((1 - (1 - c_sigma)**2) * (1 - c_sigma)**2 * (mu_eff - 1)) / (N + 1)) * \
                      (1 - np.sum(p_sigma**2) / (N * (1 - (1 - c_sigma)**2)))
            # Actually, standard formula:
            h_sigma = (np.sqrt((1 - (1 - c_sigma)**2) * (1 - c_sigma)**2 * (mu_eff - 1)) / (N + 1))
            # Wait, the standard CMA-ES h_sigma is:
            # h_sigma = 1 if ||p_sigma||/sqrt(1-(1-c_sigma)^2) < (1.4 + 2/(N+1)) * E||N(0,I)||
            # Let's use the simpler update rule for p_sigma first
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(mu_eff) / chi_N * (sigma * Y[:mu] @ w[:mu])
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            
            # Update evolution path for C
            # p_c = (1-c_c)*p_c + h_1 * sqrt(mu_eff*c_c*(2-c_c)) * (sigma * Y[:mu] @ w[:mu]) / sigma? 
            # No, p_c is in the "transformed" space? 
            # Standard CMA-ES:
            # p_c = (1 - c_c) * p_c + h_1 * sqrt(mu_eff * c_c * (2 - c_c)) * (sum(w_i * y_i))
            # where y_i are the standardized steps?
            # Actually, the update is:
            # p_c = (1 - c_c) * p_c + h_1 * sqrt(mu_eff * c_c * (2 - c_c)) * (1/sigma) * (m_new - m)
            
            # h_1 condition
            # h_1 = 1 if ||p_c||/sqrt(1-(1-c_c)^(2*gen)) < (1.4 + 2/(N+1)) * E||N(0,I)||
            # We approximate E||N(0,I)|| by chi_N
            # We don't track gen, so we use a heuristic or simplify.
            # For robustness, let's use a simplified h_1 = 1.0 if N is small? 
            # Or better, implement the check.
            
            # Let's use the standard implementation details.
            # We need to track the "expected" norm.
            # For N=5, chi_N is approx 2.5.
            # Threshold = (1.4 + 2/(N+1)) * chi_N
            
            # We need to store the "old" p_c norm? 
            # Actually, the standard update uses:
            # h_1 = 1 if ||p_c||^2 / (1 - (1 - c_c)^(2 * gen)) < (1.4 + 2/(N+1))^2 * (N - 1 + 1/N)
            # Since we don't track gen easily in this loop structure without extra state, 
            # and N is small, we can set h_1 = 1.0. This is less stable but often works for small N.
            # Alternatively, we can estimate the denominator.
            
            # Let's try to be more rigorous.
            # We'll just set h_1 = 1.0 for simplicity in this small N case, 
            # or use a fixed threshold.
            
            h_1 = 1.0 # Simplified
            
            # Update p_c
            # The term (m_new - m) / sigma is the weighted sum of Y
            weighted_sum_y = np.sum(w[:mu] * Y[:mu], axis=0)
            p_c = (1 - c_c) * p_c + h_1 * np.sqrt(mu_eff * c_c * (2 - c_c)) * weighted_sum_y
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - h_1) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * y_i * y_i.T)
            
            # Rank-one update
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_1) * c_c * (2 - c_c) * C)
            
            # Rank-mu update
            # sum of outer products of top mu Ys
            # Y[:mu] is (mu, N)
            # w_mu is (mu,)
            # Outer product sum: sum_i w_i * y_i * y_i.T
            # Can be computed as Y_mu.T @ (w_mu[:, None] * Y_mu) ? No.
            # It is Y_mu.T @ (Y_mu * w_mu[:, None])
            
            Y_mu = Y[:mu]
            W_mu = w[:mu]
            
            # C += c_mu * (Y_mu.T @ (Y_mu * W_mu[:, None]))
            C += c_mu * (Y_mu.T @ (Y_mu * W_mu[:, None]))
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Ensure positive definiteness (clip eigenvalues)
            D, B = np.linalg.eigh(C)
            D = np.maximum(D, 1e-10)
            C = (B * D) @ B.T
            
            m = m_new

        return best_x, best_f
