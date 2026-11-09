"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """CMA-ES optimizer for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # CMA-ES Parameters
        lam = int(4 + 3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Initial parameters
        m = np.zeros(dim)
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        pc = np.zeros(dim)
        
        # Update rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cc) / (dim + 1 + 3 * cc)
        c1 = (2 / ((dim + 1.3)**2 + mu_eff)) * (2 - (dim + 1.3)**-2)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Precompute
        B = np.eye(dim)
        D = np.ones(dim)
        
        while calls < budget:
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if calls >= budget:
                    break
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                f = safe_func(x)
                ys.append(y)
                xs.append(x)
                fs.append(f)
            
            if len(fs) < mu:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update mean
            m_new = m + sigma * np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
            
            # Update step size
            y_sigma = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
            norm_y_sigma = np.linalg.norm(y_sigma)
            
            # h_sigma check
            h_sigma = 1.0 if norm_y_sigma < 1.4 * np.sqrt(mu_eff) else 0.0
            
            pc = (1 - cc) * pc + h_sigma * cc * (m_new - m) / sigma
            m = m_new
            
            # Update covariance
            y_weighted_sum = np.zeros((dim, dim))
            for i in range(mu):
                y_weighted_sum += weights[i] * np.outer(ys[i], ys[i])
            
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * y_weighted_sum
            C = (C + C.T) / 2
            
            # Update B and D
            try:
                eigvals, B = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
            except:
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)
            
            # Update sigma
            sigma = sigma * np.exp(cs / (dim + 1) * (norm_y_sigma / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
