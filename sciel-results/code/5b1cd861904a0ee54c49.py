import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = int(4 + 3 * np.log(d))
        mu = lam // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)
        
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # State Initialization
        # Start at origin, assuming the shift is not too large or the function structure guides us.
        # If the shift is large, CMA-ES will still find the basin.
        rng = np.random.default_rng(0)
        x_mean = np.zeros(d)
        sigma = 1.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        while calls < budget:
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(d)
                L = np.eye(d)
                
            # Generate population
            z = rng.standard_normal((lam, d))
            y = z @ L.T
            pop = x_mean + sigma * y
            pop = np.clip(pop, -5, 5)
            
            # Evaluate population
            fvals = np.array([func(p) for p in pop])
            calls += lam
            
            # Update best if we exhausted budget
            if calls >= budget:
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = pop[min_idx].copy()
                break
            
            # Sort by fitness
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()
                
            # Update mean
            x_new = np.sum(w[:, None] * pop[:mu], axis=0)
            
            # Update path evolution strategies
            step = x_new - x_mean
            
            # ps update (in transformed space)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (L @ step)
            
            # pc update (in original space)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * step
            
            # Sigma adaptation (simplified robust rule)
            norm_ps = np.linalg.norm(ps)
            expected_norm = np.sqrt(d)
            sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / sigma - expected_norm))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Covariance matrix update
            # Rank-1
            C += c_1 * (pc @ pc.T - (1 - c_c) * (1 - 2 * c_c) * C)
            
            # Rank-mu
            y_pop = (pop[:mu] - x_mean) / sigma
            rank_mu = np.sum(w[:, None, None] * (y_pop[:, :, None] * y_pop[:, None, :]), axis=0)
            C += c_mu * rank_mu
            
            # Symmetrize and regularize
            C = (C + C.T) / 2
            C += 1e-10 * np.eye(d)
            
            x_mean = x_new

        return best_x, best_f
