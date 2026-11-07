import numpy as np
import time

class CMAES:
    def __init__(self, x0, sigma0, budget, dim, rng):
        self.dim = dim
        self.budget = budget
        self.rng = rng
        self.x = x0.copy()
        self.sigma = sigma0
        self.C = np.eye(dim)
        self.Cinv = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        
        # Parameters
        self.lam = 4 + int(3 * np.log(dim))
        if self.lam < 6: self.lam = 6
        self.mu = self.lam // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights /= np.sum(weights)
        self.weights = weights
        self.mu_eff = 1.0 / np.sum(weights**2)
        self.c_sigma = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, (self.mu_eff - 1) / (dim + 1)) + self.c_sigma
        self.c_c = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.c_1 = 2 / ((dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1.0 / self.mu_eff) / ((dim + 2)**2 + self.mu_eff))
        self.chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
        
        self.evals = 0
        self.best_x = x0.copy()
        self.best_f = np.inf
        self.f_history = []

    def step(self, func):
        if self.evals >= self.budget:
            return self.best_x, self.best_f
            
        # Sample
        B = np.linalg.cholesky(self.C).T
        z = self.rng.standard_normal((self.lam, self.dim))
        y = z @ B
        x = self.x + self.sigma * y
        f = np.array([func(xi) for xi in x])
        self.evals += self.lam
        
        # Sort
        order = np.argsort(f)
        x = x[order]
        f = f[order]
        
        # Update best
        if f[0] < self.best_f:
            self.best_f = f[0]
            self.best_x = x[0].copy()
            
        # Update mean
        self.x = self.x + self.sigma * (self.weights[:self.mu] @ x[:self.mu])
        
        # Update evolution paths
        yw = (self.weights[:self.mu] @ x[:self.mu])
        self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (B @ yw) / self.sigma
        if np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.c_sigma)**(2 * (self.evals + 1))) < (1.4 + 1 / self.dim) * self.chiN:
            self.pc = (1 - self.c_c) * self.pc + np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * yw
        else:
            self.pc = (1 - self.c_c) * self.pc
            
        # Update covariance
        delta_diag = (np.linalg.norm(self.ps)**2 / self.dim - 1) * self.c_sigma * (2 - self.c_sigma)
        self.C = (1 + self.c_1 + self.c_mu - self.c_1 * self.trace_C_inv_C) * self.C \
                 + self.c_1 * (np.outer(self.pc, self.pc) - self.C) \
                 + self.c_mu * sum(w * np.outer(x[i], x[i]) for i, w in enumerate(self.weights[:self.mu])) \
                 + delta_diag * self.C
        # Ensure symmetry and positive definite
        self.C = (self.C + self.C.T) / 2
        try:
            self.Cinv = np.linalg.inv(self.C)
            self.trace_C_inv_C = np.trace(self.C @ self.Cinv)
        except:
            self.Cinv = np.eye(self.dim)
            self.trace_C_inv_C = self.dim
            
        # Update sigma
        self.sigma *= np.exp(self.c_sigma / self.d_sigma * (np.linalg.norm(self.ps) / self.chiN - 1))
        
        # Clip sigma
        if self.sigma < 1e-10:
            self.sigma = 1e-10
        if self.sigma > 10:
            self.sigma = 10

        return self.best_x, self.best_f

    def run(self, func):
        while self.evals < self.budget:
            self.step(func)
            if self.sigma < 1e-8:
                break
        return self.best_x, self.best_f

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        start_time = time.time()
        
        # Strategy: 3 restarts
        # 1. Center, large sigma
        # 2. Random, medium sigma
        # 3. Best found, small sigma (polish)
        
        configs = [
            (np.zeros(self.dim), 2.5, int(self.budget * 0.4)),
            (rng.uniform(-4, 4, self.dim), 1.5, int(self.budget * 0.4)),
            (None, 0.5, self.budget * 0.2), # None means use best so far
        ]
        
        for x0, sigma0, alloc in configs:
            if x0 is None:
                if best_x is not None:
                    x0 = best_x + rng.normal(0, 0.1, self.dim)
                else:
                    x0 = rng.uniform(-5, 5, self.dim)
            
            # Check time
            if time.time() - start_time > 280:
                break
                
            cma = CMAES(x0, sigma0, alloc, self.dim, rng)
            bx, bf = cma.run(func)
            
            if bf < best_f:
                best_f = bf
                best_x = bx
        
        if best_x is None:
             # Fallback if something went wrong
             best_x = np.zeros(self.dim)
             best_f = func(best_x)
             
        return best_x, best_f
