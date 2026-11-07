import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.call_count = 0
        self.best_f = np.inf
        self.best_x = None
        self.start_time = time.time()
        
        time_limit = 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def _safe_eval(x):
            if self.call_count >= self.budget:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            
            self.call_count += 1
            if np.isfinite(fx) and fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        # Phase 1: Global Search using CMA-ES
        # CMA-ES is robust for global optimization and handles the rotated/shifted landscape well.
        # We allocate roughly 60% of the budget to this phase.
        
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / sum(weights**2)
        
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = 2.0 / ((dim + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0)**2 + mueff))
        
        # Initial point: center of the box or random
        x_mean = rng.uniform(-5, 5, dim)
        C = np.eye(dim)
        sigma = 2.0 # Wide initial step to explore the 10-unit box
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        D, V = np.linalg.eigh(C)
        
        chiN = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))
        
        cma_budget_limit = int(budget * 0.6)
        
        while self.call_count < cma_budget_limit:
            if time.time() - self.start_time > time_limit - 50: # Leave 50s for local
                break
                
            # Check if we can afford a full generation
            if self.call_count + lambda_ > cma_budget_limit:
                break
                
            ys = []
            xs = []
            fs = []
            
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = V @ (np.sqrt(D) * z)
                x = x_mean + sigma * y
                x = np.clip(x, -5, 5)
                
                fx = _safe_eval(x)
                ys.append(y)
                xs.append(x)
                fs.append(fx)
                
                if self.call_count >= cma_budget_limit:
                    break
            
            if len(ys) < 2:
                break
                
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * xs[i]
                
            # Update ps
            # inv_sqrt_C = V @ diag(1/sqrt(D)) @ V.T
            inv_sqrt_C = V @ (np.diag(1.0 / np.sqrt(D))) @ V.T
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (inv_sqrt_C @ (x_new - x_mean)) / sigma
            
            # Update pc
            pc = (1.0 - cc) * pc + cc * (x_new - x_mean) / sigma
            
            # Update C
            rank1 = ps @ ps.T + (cs * (2.0 - cs)) * (np.dot(ps, ps) / dim - 1.0) * C
            rankmu = np.zeros((dim, dim))
            for i in range(mu):
                rankmu += weights[i] * np.outer(ys[i], ys[i])
                
            C = (1.0 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma = sigma * np.exp(cs / (2.0 * dim) * (np.linalg.norm(ps) / chiN - 1.0))
            
            # Update mean
            x_mean = x_new
            
            # Update eigen decomposition
            D, V = np.linalg.eigh(C)
            D = np.maximum(D, 1e-12)
            
            if sigma < 1e-8:
                break

        # Phase 2: Local Refinement using Nelder-Mead
        # Nelder-Mead is derivative-free and often effective for refining the solution 
        # found by CMA-ES, especially in the narrow valley of Rosenbrock.
        
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            _safe_eval(self.best_x)
            
        if self.call_count < budget - 10:
            x_start = self.best_x.copy()
            
            # We will run Nelder-Mead manually to strictly control the call count and avoid
            # SciPy's internal overhead or potential budget overruns if not careful.
            # However, implementing NM from scratch is error-prone.
            # Let's use a simple (1+1)-ES with adaptive step size as a robust fallback
            # which is very effective for smooth functions like Rosenbrock once in the basin.
            
            x = x_start
            fx = self.best_f
            sigma_es = 0.5 # Start with a moderate step
            
            while self.call_count < budget - 5:
                if time.time() - self.start_time > time_limit:
                    break
                    
                y = x + sigma_es * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = _safe_eval(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma_es *= 1.05 # Slight increase
                else:
                    sigma_es *= 0.8  # Decrease step size on failure
                    
                sigma_es = np.clip(sigma_es, 1e-6, 5.0)
                
                # Optional: If sigma is very small and we haven't improved for a while, 
                # we could try a larger jump, but for Rosenbrock, local refinement is key.
                
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x

        return self.best_x, self.best_f

    def _safe_eval(self, func, x):
        # This method is kept for compatibility but the inner __call__ uses its own closure
        # to ensure strict budget counting within the specific run.
        # If called externally, it behaves as before.
        if self.call_count >= self.budget:
            raise ValueError("Budget Exceeded")
        
        x = np.clip(x, -5, 5)
        f = func(x)
        self.call_count += 1
        
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            
        return f
