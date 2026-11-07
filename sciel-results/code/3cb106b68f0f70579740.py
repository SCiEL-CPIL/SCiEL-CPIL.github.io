import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 9
        
        # Initial mean
        m = np.zeros(n)
        
        # Initial step size
        sigma = 1.0
        
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Path variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Evolution paths parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        best_x = m.copy()
        best_f = func(m)
        
        calls = 1
        
        while calls < budget:
            # Generate population
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.diag(1.0 / np.sqrt(eigvals))
            
            Z = np.random.randn(lam, n)
            Y = Z @ (sigma * D @ eigvecs.T)
            X = np.array([m + y for y in Y])
            
            num_eval = min(lam, budget - calls)
            
            F = []
            X_eval = []
            X_clipped = []
            for i in range(num_eval):
                xi = X[i]
                xi_clip = np.clip(xi, -5, 5)
                fi = func(xi_clip)
                F.append(fi)
                X_eval.append(xi)
                X_clipped.append(xi_clip)
                calls += 1
                if fi < best_f:
                    best_f = fi
                    best_x = xi_clip.copy()
            
            if num_eval < lam:
                break
                
            F = np.array(F)
            X_clipped = np.array(X_clipped)
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X_clipped[order]
            F_sorted = F[order]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update sigma
            diff = m - m_old
            # v = C^-1/2 (m - m_old)
            v = eigvecs @ (1.0 / np.sqrt(eigvals)) @ (eigvecs.T @ diff)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (v / sigma)
            # Threshold for hsig
            # Standard formula: (||p_sigma|| / sqrt(1-(1-c_sigma)^{2*1})) < (1.4 + 2/(n+1)) * sqrt((mueff+1)/(n+1))
            # The term 2*1 in exponent is for 1 step.
            thresh = (1.4 + 2.0 / (n + 1)) * np.sqrt((mueff + 1) / (n + 1))
            denom = np.sqrt(1.0 - (1.0 - c_sigma)**2)
            hsig = (np.linalg.norm(p_sigma) / denom) < thresh
            
            # Update p_c
            if hsig:
                p_c = (1 - c_c) * p_c + c_c * (v / sigma)
            
            # Update C
            H = np.outer(diff, diff)
            
            diffs = X_sorted[:mu] - m_old
            # Whitened diffs
            Y_proj = eigvecs.T @ diffs.T
            Y_proj = Y_proj / np.sqrt(eigvals)[:, None]
            W = eigvecs @ Y_proj
            H_mu = W @ np.diag(weights) @ W.T
            
            C = (1 + c_1 + c_mu - c_c) * C + c_1 * H + c_mu * H_mu
            C = (C + C.T) / 2
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

        return best_x, best_f
