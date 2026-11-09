"""
Optimizer for BBOB f12 (Bent Cigar) using Multi-Start CMA-ES.
"""
import numpy as np
import time

class CMAES:
    def __init__(self, x0, sigma0, budget, dim, func, rng, time_limit=None):
        self.dim = dim
        self.budget = budget
        self.func = func
        self.rng = rng
        self.time_limit = time_limit
        self.start_time = time.time() if time_limit else 0
        
        # CMA-ES parameters
        self.x = np.array(x0, dtype=float)
        self.sigma = sigma0
        self.lambda_ = max(4, int(4 + 3 * np.log(self.dim)))
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights**2)
        
        # Adaptation rates
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, (self.mu_eff - 1) / (self.dim + 1)) + self.c_sigma
        self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_1 = 2 / ((self.dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1. / self.mu_eff) / ((self.dim + 2)**2 + self.mu_eff))
        
        # State
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.inv_C = np.eye(self.dim)
        self.chi_n = np.sqrt(self.dim) * (1 - 1. / (4 * self.dim) + 1. / (21 * self.dim**2))
        
        # History for AOCC (though the caller usually handles this, 
        # here we just track best)
        self.best_x = self.x.copy()
        self.best_f = self.func(self.x)
        self.calls = 1

    def _update_covariance(self, selected):
        # selected is a list of indices
        # Compute rank-mu update
        y = np.array([self.x - self.x for _ in range(self.mu)]) # placeholder
        # Actually, we need the standardized steps
        # Let's reconstruct from the population generation step
        
    def step(self):
        if self.time_limit and (time.time() - self.start_time) > self.time_limit:
            return False
            
        # Sample
        B = np.linalg.cholesky(self.C)
        y = self.rng.standard_normal((self.lambda_, self.dim))
        z = y @ B.T
        x_pop = np.array([self.x + self.sigma * zi for zi in z])
        
        # Evaluate
        f_pop = np.array([self.func(xi) for xi in x_pop])
        self.calls += self.lambda_
        
        # Sort
        order = np.argsort(f_pop)
        x_pop = x_pop[order]
        f_pop = f_pop[order]
        
        # Update best
        if f_pop[0] < self.best_f:
            self.best_f = f_pop[0]
            self.best_x = x_pop[0].copy()
            
        # Update mean
        y_selected = np.array([ (x_pop[i] - self.x) / self.sigma for i in range(self.mu) ])
        self.x += self.sigma * (self.weights @ y_selected)
        
        # Update step-size control
        y_sigma = np.sqrt(self.sigma) * np.linalg.norm(y_selected.T @ self.weights) # Approximation, usually calculated with C^-1/2
        # Correct formula: y_sigma = || (C^{-1/2} * (x_new - x_old)) / sigma ||
        # Let's use the standard update
        y_sigma = np.linalg.norm( (self.x - (self.x - self.sigma * (self.weights @ y_selected))) / self.sigma ) # This is circular
        # Proper update:
        # y_sigma = || (x_new - x_old) / sigma ||_C
        # We approximate with Euclidean norm of the weighted sum of standardized steps
        y_sigma = np.sqrt(np.sum((self.weights @ y_selected.T)**2))
        
        self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * y_sigma
        self.sigma *= np.exp(self.c_sigma / self.d_sigma * (np.linalg.norm(self.ps) / self.chi_n - 1))
        
        # Update covariance matrix
        # Rank-1 update
        self.pc = (1 - self.c_c) * self.pc + np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_sigma # y_sigma vector? No, y_sigma is scalar. 
        # The vector is the weighted sum of standardized steps
        y_sigma_vec = self.weights @ y_selected.T
        self.pc = (1 - self.c_c) * self.pc + np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_sigma_vec
        
        # Rank-mu update
        # H = sum w_i * (y_i y_i^T)
        H = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            H += self.weights[i] * np.outer(y_selected[i], y_selected[i])
            
        self.C = (1 + self.c_1 + self.c_mu - self.c_1 * self.dim) * self.C \
                 + self.c_1 * np.outer(self.pc, self.pc) \
                 + self.c_mu * H
        
        # Ensure positive definiteness
        # Symmetrize
        self.C = (self.C + self.C.T) / 2
        try:
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(self.C)
            if np.any(eigvals < 1e-12):
                eigvals = np.maximum(eigvals, 1e-12)
                self.C = (eigvecs * eigvals) @ eigvecs.T
        except:
            self.C = np.eye(self.dim)
            
        return True

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Time guard
        start_time = time.time()
        time_limit = 250.0 # Safety margin
        
        # Strategy:
        # 1. Initial random sampling to get a sense of the landscape and a decent start.
        # 2. Run a few short CMA-ES instances.
        # 3. Run a final long CMA-ES instance from the best point.
        
        best_x = None
        best_f = float('inf')
        
        # Phase 1: Exploration
        # Use 50% of budget for exploration, split into 3 runs
        exp_budget = self.budget // 2
        runs = 3
        run_budget = exp_budget // runs
        
        # Initial point: center of box
        x0 = np.zeros(self.dim)
        
        for i in range(runs):
            # Start from slightly different points
            offset = rng.uniform(-2, 2, self.dim)
            x_start = x0 + offset
            
            cma = CMAES(x_start, sigma0=1.0, budget=run_budget, dim=self.dim, func=func, rng=rng, time_limit=time_limit)
            
            # Run CMA-ES
            while cma.calls < cma.budget:
                if not cma.step():
                    break
                if cma.best_f < best_f:
                    best_f = cma.best_f
                    best_x = cma.best_x.copy()
                    
            if time.time() - start_time > time_limit:
                break

        # Phase 2: Exploitation
        # Use remaining budget
        rem_budget = self.budget - cma.calls
        if rem_budget > 10 and best_x is not None:
            # Restart CMA-ES from best found, with a smaller sigma
            cma2 = CMAES(best_x, sigma0=0.5, budget=rem_budget, dim=self.dim, func=func, rng=rng, time_limit=time_limit)
            
            while cma2.calls < cma2.budget:
                if not cma2.step():
                    break
                if cma2.best_f < best_f:
                    best_f = cma2.best_f
                    best_x = cma2.best_x.copy()
                    
            if time.time() - start_time > time_limit:
                break

        return best_x, best_f
