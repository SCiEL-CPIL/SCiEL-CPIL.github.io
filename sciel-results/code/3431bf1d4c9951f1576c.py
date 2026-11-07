import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Parameters for CMA-ES
        # Popsize lambda
        lam = 4 + int(3 * np.log(n)) if n > 1 else 2
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        # Start at 0, which is in the center of the box and likely close to the shifted optimum
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Pre-compute constants
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # We need to ensure we don't exceed budget
        # Each generation uses 'lam' evaluations
        # Number of generations
        n_gen = budget // lam
        rem = budget % lam
        
        self.best_x = x_mean.copy()
        self.best_f = func(self.best_x)
        self.calls_made += 1
        
        if self.calls_made >= budget:
            return self.best_x, self.best_f

        for g in range(n_gen):
            # Check time limit (safety margin: stop if less than 5s remaining)
            if time.time() - self.start_time > 325:
                break
            
            # Check budget: need at least lam more calls
            if self.calls_made + lam > budget:
                break
                
            # Generate candidates
            # Pre-compute Cholesky of C for efficiency
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigendecomposition
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-12)
                L = (vecs * np.sqrt(vals)) @ vecs.T
            
            ys = np.zeros((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                ys[i] = x_mean + sigma * L @ z
                ys[i] = np.clip(ys[i], -5, 5)
            
            # Evaluate
            fs = np.array([func(y) for y in ys])
            self.calls_made += lam
            
            # Sort indices
            order = np.argsort(fs)
            
            # Update best
            if fs[order[0]] < self.best_f:
                self.best_f = fs[order[0]]
                self.best_x = ys[order[0]].copy()
                
            # Compute weighted sum of selected points (in standard normal space)
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += w[i] * (ys[order[i]] - x_mean) / sigma
            
            # Update covariance adaptation path
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update step-size control path
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            
            # Update sigma
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)
            
            # Update mean
            x_mean = x_mean + sigma * y_w
            x_mean = np.clip(x_mean, -5, 5)
            
            # Update Covariance C
            # Delta H correction
            delta_h = (1 - np.exp(-2 * (1 - c_c) * (g + 1)))
            h_sig = 1 if norm_ps < (1.4 + 1 / c_sigma) * chi_n else 0
            
            # Compute rank-mu update using old x_mean
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                diff = (ys[order[i]] - x_mean) / sigma
                rank_mu += w[i] * np.outer(diff, diff)
            
            # C = (1 - c_1 - c_mu) * C + c_1 * (outer(pc, pc) + (1-h_sig)*c_sigma*(2-c_sigma)*(outer(ps, ps) - trace(C)*I)) + c_mu * rank_mu
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sig) * c_sigma * (2 - c_sigma) * (np.outer(ps, ps) - np.trace(C) * np.eye(n))) + c_mu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Check budget
            if self.calls_made >= budget:
                break

        # Use remaining budget for single step (1+1) or just return
        while self.calls_made < budget:
            if time.time() - self.start_time > 325:
                break
            # Simple (1+1) step using current sigma and C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-12)
                L = (vecs * np.sqrt(vals)) @ vecs.T
            
            z = rng.standard_normal(n)
            y = x_mean + sigma * L @ z
            y = np.clip(y, -5, 5)
            fy = func(y)
            self.calls_made += 1
            if fy < self.best_f:
                self.best_f = fy
                self.best_x = y.copy()
                sigma *= 1.1
            else:
                sigma *= 0.9
                sigma = np.clip(sigma, 1e-12, 10.0)

        return self.best_x, self.best_f
