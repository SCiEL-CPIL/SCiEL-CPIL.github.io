import numpy as np

class Optimizer:
    """CMA-ES Optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # CMA-ES parameters
        self.lambda_ = 4 + int(3 * np.log(dim))
        if self.lambda_ < 2:
            self.lambda_ = 2
        self.mu_eff = int(np.floor(self.lambda_ / 2))
        
        # Weights
        self.weights = np.log(self.lambda_ / 2 + 0.5) - np.log(1 + np.arange(self.mu_eff))
        self.weights /= self.weights.sum()
        self.mu_eff_weighted = np.sum(self.weights**2)
        
        # Adaptation parameters
        self.c1 = (4 - self.mu_eff_weighted + 1) / ((dim + 1.3)**2 + self.mu_eff_weighted)
        self.cmua = 2 * (self.mu_eff_weighted - 2 + 1/self.mu_eff_weighted) / ((dim + 2)**2 + self.mu_eff_weighted)
        self.cs = (self.mu_eff_weighted + 2) / (dim + self.mu_eff_weighted + 5)
        self.ds = 1 + 2 * max(0, np.sqrt((self.mu_eff_weighted - 1) / (dim + 1)) - 1) + self.cs
        self.c_sigma = (self.cs + 1) / (dim + 4)
        self.d_sigma = 1 + 2 * max(0, np.sqrt(self.mu_eff_weighted / dim) - 1) + self.c_sigma
        self.chi_lambda = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Initial state
        m = np.zeros(dim)  # Mean
        sigma = 2.0        # Initial step size
        C = np.eye(dim)    # Covariance matrix
        L = np.linalg.cholesky(C)
        
        # Paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Best solution found
        best_x = None
        best_f = np.inf
        
        # Generation loop
        while budget >= self.lambda_:
            # Sample population
            X = np.zeros((self.lambda_, dim))
            for i in range(self.lambda_):
                z = rng.standard_normal(dim)
                X[i] = m + sigma * (L @ z)
                
            # Evaluate
            F = np.array([func(x) for x in X])
            budget -= self.lambda_
            
            # Track best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()
                
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            
            # Select top mu
            mu = self.mu_eff
            X_sel = X_sorted[:mu]
            w = self.weights
            
            # Update mean
            m_new = np.sum(w[:, np.newaxis] * X_sel, axis=0)
            
            # Update step size (sigma)
            # p_sigma update
            # Solve L y = (m_new - m) / sigma
            try:
                y_sigma = np.linalg.solve(L, (m_new - m) / sigma)
            except np.linalg.LinAlgError:
                y_sigma = np.zeros(dim)
                
            p_sigma = (1 - self.cs) * p_sigma + self.cs * np.sqrt(self.mu_eff_weighted) * y_sigma
            
            # Check if p_sigma is within the trust region
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(self.c_sigma / self.ds * (norm_p_sigma / self.chi_lambda - 1))
            
            # Update covariance matrix (C)
            # p_c update
            p_c = (1 - self.c1) * p_c + np.sqrt(self.c1 * (2 - self.c1) * self.mu_eff_weighted) * (m_new - m) / sigma
            
            # Rank-one update
            C = (1 + self.c1 * self.cmua - self.c1) * C + self.c1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            # y_i = (x_i - m) / sigma
            Y = (X_sel - m) / sigma
            C += self.cmua * np.sum(w[:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', Y, Y), axis=0)
            
            # Update mean
            m = m_new
            
            # Update Cholesky factor
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-4 * np.eye(dim)
                L = np.linalg.cholesky(C)
                
        return best_x, best_f
