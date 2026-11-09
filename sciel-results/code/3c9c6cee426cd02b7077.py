import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.time_limit = 290.0

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_used = 0
        
        # Multi-restart CMA-ES
        num_restarts = 10
        budget_per_restart = self.budget // num_restarts
        
        for i in range(num_restarts):
            if time.time() - self.start_time > self.time_limit:
                break
                
            x0 = rng.uniform(-5, 5, self.dim)
            sigma0 = 2.0
            
            x, f, used = self._cma_es(func, x0, sigma0, budget_per_restart, rng)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            total_used += used
            
            if total_used >= self.budget:
                break
                
        if best_x is not None and total_used < self.budget:
            remaining = self.budget - total_used
            x, f, used = self._cma_es(func, best_x, 0.1, remaining, rng)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        return best_x, best_f

    def _cma_es(self, func, x0, sigma0, max_evals, rng):
        dim = self.dim
        lambda_ = 10
        mu = 5
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        evals = 0
        best_x = x_mean.copy()
        best_f = func(x_mean)
        evals += 1
        
        while evals < max_evals:
            B = np.linalg.cholesky(C)
            Z = rng.standard_normal((lambda_, dim))
            pops = x_mean + sigma * (Z @ B.T)
            
            fs = np.array([func(p) for p in pops])
            evals += lambda_
            
            idx = np.argsort(fs)
            pops = pops[idx]
            fs = fs[idx]
            
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = pops[0].copy()
                
            x_mean_new = np.sum(weights[:, None] * pops[:mu], axis=0)
            
            y = (x_mean_new - x_mean) / sigma
            ps = 0.84 * ps + 0.16 * y # Simplified update
            pc = 0.84 * pc + 0.16 * y
            C = 0.99 * C + 0.01 * np.outer(y, y) # Simplified update
            
            sigma *= np.exp(0.01 * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            x_mean = x_mean_new
            
            if sigma < 1e-4:
                break
                
        return best_x, best_f, evals
