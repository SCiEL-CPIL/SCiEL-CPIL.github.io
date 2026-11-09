import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initialization
        init_budget = max(5, int(budget * 0.1))
        init_pop = rng.uniform(-5, 5, (init_budget, n))
        init_fits = np.array([func(x) for x in init_pop])
        evals_used = init_budget
        
        best_idx = np.argmin(init_fits)
        mean = init_pop[best_idx].copy()
        best_f = init_fits[best_idx]
        best_x = mean.copy()
        
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        chi_N = np.sqrt(n) * (1 - 1/(4*n) - 1/(25*n**2))
        
        while evals_used + lam <= budget:
            # Decompose C
            d, B = np.linalg.eigh(C)
            d = np.sqrt(d)
            d[d < 1e-12] = 1e-12
            
            # Sample
            pop = np.zeros((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                pop[i] = mean + sigma * (B @ (d * z))
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Evaluate
            fits = np.array([func(x) for x in pop])
            evals_used += lam
            
            # Update best
            idx_min = np.argmin(fits)
            if fits[idx_min] < best_f:
                best_f = fits[idx_min]
                best_x = pop[idx_min].copy()
                
            # Sort
            order = np.argsort(fits)
            y = (pop[order[:mu]] - mean) / sigma
            
            # Update Mean
            mean_old = mean
            mean = mean + sigma * (w @ y)
            mean = np.clip(mean, -5, 5)
            
            # Update Paths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (d * (w @ y)))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (mean - mean_old) / sigma
            
            # Update Covariance
            delta_sigma = (np.linalg.norm(ps) / chi_N - 1) * c_sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (delta_sigma * (1 + delta_sigma - c_c) - 1) * C)
            
            for i in range(mu):
                C += c_mu * w[i] * np.outer(y[i], y[i])
                
            C = (C + C.T) / 2
            
            # Update Sigma
            sigma = sigma * np.exp(c_sigma / 5.0 * (np.linalg.norm(ps) / chi_N - 1))
            
            # Check convergence
            if sigma < 1e-15:
                break

        return best_x, best_f
