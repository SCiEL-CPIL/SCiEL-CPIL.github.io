import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        lam = max(4, 4 + int(3 * np.log(n)))
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 1.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        
        calls_used = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        max_gens = budget // lam + 1
        
        for gen in range(max_gens):
            # Generate and evaluate offspring
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if calls_used >= budget:
                    break
                z = rng.standard_normal(n)
                # Sample y ~ N(0, C) using Cholesky
                try:
                    L = np.linalg.cholesky(C + 1e-8 * np.eye(n))
                    y = L @ z
                except np.linalg.LinAlgError:
                    y = z
                x = x_mean + sigma * y
                x = np.clip(x, -5, 5)
                f = func(x)
                calls_used += 1
                ys.append(y)
                xs.append(x)
                fs.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(ys) < 2:
                break
            
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            xs = xs[order]
            fs = fs[order]
            
            # Update mean
            x_mean_new = np.sum(w[:, None] * xs[:mu], axis=0)
            
            # Update evolution path for covariance adaptation
            y_mean = w @ ys
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_mean / sigma)
            
            # Heaviside function for h
            h = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1-c_sigma)**(2*(gen+1))) < (1.4 + 1/n) * chi_n else 0
            
            # Update covariance matrix
            C = (1 + c_1 + c_mu - c_1 - np.trace(C)) * C + \
                c_1 * (np.outer(pc, pc) + (1 - h) * (1 - np.exp(-2*c_c)) * C) + \
                c_mu * np.sum(w[:, None, None] * np.outer(ys[:mu] - y_mean, ys[:mu] - y_mean), axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update step-size (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_mean / sigma)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            x_mean = x_mean_new
            
            # Stop conditions
            if sigma < 1e-10 or calls_used >= budget:
                break
        
        return best_x, best_f
