import numpy as np

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        self.lam = 4 + int(3 * np.log(self.dim))
        self.mu = self.lam // 2
        
        # Initial mean: center of the box
        self.m = np.zeros(self.dim)
        
        # Initial sigma: 2.0 is a good heuristic for [-5, 5] box
        self.sigma = 2.0
        
        # Initial covariance matrix: Identity
        self.C = np.eye(self.dim)
        
        # Precompute weights
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights**2)
        
        # CMA-ES parameters
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.c1 = 2.0 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmua = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((self.dim + 2.0)**2 + self.mueff))
        
        # Normalization factors
        self.chi_n = np.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim**2))
        self.dtsigma = self.dim / self.cs + 2.0
        
        # Evolution paths
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        
        # B and D for Cholesky decomposition
        B, D = np.linalg.cholesky(self.C)
        
        best_x = None
        best_f = float('inf')
        
        calls_made = 0
        
        while calls_made < self.budget:
            # Generate population
            Y = np.zeros((self.lam, self.dim))
            X = np.zeros((self.lam, self.dim))
            F = np.zeros(self.lam)
            
            for i in range(self.lam):
                z = rng.standard_normal(self.dim)
                Y[i] = self.sigma * (B @ z)
                X[i] = self.m + Y[i]
                # Clip to box [-5, 5]
                X[i] = np.clip(X[i], -5, 5)
                F[i] = func(X[i])
                calls_made += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
                
                if calls_made >= self.budget:
                    break
            
            if calls_made >= self.budget:
                break
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            Y_sorted = Y[order]
            
            # Update mean
            self.m = np.sum(self.weights[:, np.newaxis] * X_sorted[:self.mu], axis=0)
            
            # Update evolution paths
            y_mean = np.sum(self.weights[:, np.newaxis] * Y_sorted[:self.mu], axis=0)
            self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_mean / self.sigma
            self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * y_mean
            
            # Update sigma
            self.sigma *= np.exp(self.cs / self.dtsigma * (np.linalg.norm(self.ps) / self.chi_n - 1.0))
            
            # Update covariance matrix
            # C = (1 - c1 - cmua + cc(2 - cc)) * C + c1 * pc pc^T + cmua * sum wi yi yi^T
            C_new = (1.0 - self.c1 - self.cmua + self.cc * (2.0 - self.cc)) * self.C
            C_new += self.c1 * np.outer(self.pc, self.pc)
            
            sum_outer = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                sum_outer += self.weights[i] * np.outer(Y_sorted[i], Y_sorted[i])
            C_new += self.cmua * sum_outer
            
            self.C = C_new
            
            # Ensure positive semi-definite and recompute Cholesky
            self.C = (self.C + self.C.T) / 2.0
            try:
                B, D = np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                # Fallback: reset C to identity scaled
                self.C = np.eye(self.dim) * np.trace(self.C) / self.dim
                B, D = np.linalg.cholesky(self.C)

        return best_x, best_f
