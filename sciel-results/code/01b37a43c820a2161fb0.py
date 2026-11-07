"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with adaptive step size and multiple restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(123)
        n = self.dim
        total_budget = self.budget
        calls = [0]
        best_x = None
        best_f = np.inf

        def eval_counted(x):
            if calls[0] >= total_budget:
                return np.inf
            calls[0] += 1
            return func(x)

        def cma_es(x0, sigma0, budget_limit):
            nonlocal best_x, best_f
            n = self.dim
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            
            cs = 1 / (n + 3)
            c1 = 2 / ((n + 1.3) ** 2 + 1)
            cmu = min(1 - c1, 2 / ((n + 2) ** 2 + 5))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            pop_size = 4 + int(3 * np.log(n))
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            
            while calls[0] < budget_limit:
                if calls[0] >= total_budget:
                    break
                
                # Generate offspring
                offspring = []
                for i in range(pop_size):
                    if calls[0] >= total_budget:
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * (C @ z)
                    y = np.clip(y, -5, 5)
                    fy = eval_counted(y)
                    offspring.append((y, fy))
                
                if len(offspring) < mu:
                    break
                
                # Sort
                offspring.sort(key=lambda t: t[1])
                
                # Update best
                if offspring[0][1] < best_f:
                    best_f = offspring[0][1]
                    best_x = offspring[0][0].copy()
                
                # Weighted mean
                x_new = np.zeros(n)
                for i in range(mu):
                    x_new += weights[i] * offspring[i][0]
                
                # Update sigma
                z_mean = np.zeros(n)
                for i in range(mu):
                    try:
                        z_i = np.linalg.solve(C, (offspring[i][0] - x) / sigma)
                        z_mean += weights[i] * z_i
                    except np.linalg.LinAlgError:
                        z_i = (offspring[i][0] - x) / sigma
                        z_mean += weights[i] * z_i
                
                success = np.linalg.norm(z_mean) / chi_n
                sigma *= np.exp(cs * (success - 1))
                sigma = np.clip(sigma, 1e-12, 5.0)
                
                # Update C
                delta_x = x_new - x
                pc = (1 - cs) * pc + cs / np.sqrt(cs * (2 - cs)) * (delta_x / sigma)
                
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (cs * (2 - cs)) * np.eye(n))
                for i in range(mu):
                    try:
                        z_i = np.linalg.solve(C, (offspring[i][0] - x) / sigma)
                        C += cmu * weights[i] * np.outer(z_i, z_i)
                    except np.linalg.LinAlgError:
                        pass
                
                C = (C + C.T) / 2
                x = x_new
            
            return x, best_f

        # Multiple restarts
        n_restarts = 3
        budget_per_restart = total_budget // n_restarts
        
        for r in range(n_restarts):
            if calls[0] >= total_budget:
                break
            x0 = rng.uniform(-4, 4, n)
            sigma0 = 1.5
            cma_es(x0, sigma0, calls[0] + budget_per_restart)
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = eval_counted(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
