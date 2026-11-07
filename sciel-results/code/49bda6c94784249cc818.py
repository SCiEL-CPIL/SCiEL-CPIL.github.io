import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x_global = None
        best_f_global = np.inf
        
        # 3 independent CMA-ES runs
        num_restarts = 3
        budget_per_restart = total_budget // num_restarts
        # Give the last one the remainder
        budgets = [budget_per_restart] * (num_restarts - 1) + [total_budget - budget_per_restart * (num_restarts - 1)]
        
        for k in range(num_restarts):
            budget = budgets[k]
            
            # CMA-ES Parameters
            lambda_ = 4 + int(3 * np.log(n))
            mu = lambda_ // 2
            
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # Initial state: Random center in [-4, 4] to avoid boundary issues
            x_mean = rng.uniform(-4, 4, size=n)
            sigma = 1.5
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            try:
                chol_C = np.linalg.cholesky(C)
            except:
                chol_C = np.eye(n)
            
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            calls = 0
            best_x_local = x_mean.copy()
            best_f_local = func(best_x_local)
            calls += 1
            
            while calls < budget:
                # Generate population
                z = rng.standard_normal((lambda_, n))
                pop = x_mean + sigma * (z @ chol_C.T)
                pop = np.clip(pop, -5, 5)
                
                # Evaluate
                fvals = np.array([func(x) for x in pop])
                calls += lambda_
                
                # Update best
                idx_min = np.argmin(fvals)
                if fvals[idx_min] < best_f_local:
                    best_f_local = fvals[idx_min]
                    best_x_local = pop[idx_min].copy()
                
                # Sort
                order = np.argsort(fvals)
                
                # Update mean
                x_mean_new = np.sum(w[:, np.newaxis] * pop[order[:mu]], axis=0)
                
                # Update sigma
                x_sigma = (x_mean_new - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * x_sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1) / lambda_))) < (1.4 + 2 / (n + 1)) * chi_n
                sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update pc
                if hsig:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * x_sigma
                
                # Update C
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) + (hsig - 1) * c_c * (2 - c_c) * C)
                diff = pop[order[:mu]] - x_mean
                C += c_mu * np.sum(w[:, np.newaxis, np.newaxis] * np.einsum('ij,ik->ijk', diff, diff), axis=0)
                
                C = (C + C.T) / 2
                
                try:
                    chol_C = np.linalg.cholesky(C)
                except:
                    C += 1e-6 * np.eye(n)
                    chol_C = np.linalg.cholesky(C)
                
                x_mean = x_mean_new
                
            if best_f_local < best_f_global:
                best_f_global = best_f_local
                best_x_global = best_x_local.copy()
        
        return best_x_global, best_f_global
