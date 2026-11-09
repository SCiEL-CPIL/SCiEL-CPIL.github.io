"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: CMA-ES for exploration, (1+1)-ES for exploitation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        
        # Phase 1: CMA-ES for broad exploration
        lambda_ = 4 + 3 * int(np.log(dim))  # population size ~ 4 + 3*1 = 7 for dim=5
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + (4 + dim / (dim + 4)))
        cs = (dim + 1.7) / (dim + 9)
        c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - cc)
        cmu = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (cc - 1 + 2 * mu_eff / lambda_)
        
        # Initialize
        x = np.zeros(dim)
        sigma = 2.0
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        
        calls_made = 0
        best_x = x.copy()
        best_f = np.inf
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Initial population
        pop = [x + sigma * np.random.standard_normal(dim) for _ in range(lambda_)]
        pop_f = [evaluate(np.clip(p, -5, 5)) for p in pop]
        
        # Sort
        idx = np.argsort(pop_f)
        pop = [pop[i] for i in idx]
        pop_f = [pop_f[i] for i in idx]
        
        # CMA-ES loop for first 70% of budget
        cma_budget = int(budget * 0.7)
        while calls_made < cma_budget and time.time() < deadline:
            # Update mean
            m_old = x.copy()
            x = sum(weights[i] * pop[i] for i in range(mu))
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * B.T @ ((x - m_old) / (sigma * D))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made + 1) // lambda_))) / \
                   np.sqrt(mu_eff) * (1 - cc) < 1.4 + 2 / (dim + 1)
            if hsig:
                pc = (1 - cc) * pc + cc * (x - m_old) / sigma
            else:
                pc = (1 - cc) * pc
            
            # Update covariance matrix
            delta_h = (2 * hsig - 1) * cc * (1 - cc)
            C = (1 + c1 * delta_h - c1) * C + c1 * np.outer(pc, pc) + cmu * sum(
                weights[i] * np.outer(pop[i] - m_old, pop[i] - m_old) for i in range(mu)
            )
            
            # Eigendecomposition (periodic to save time)
            if calls_made % 10 == 0:
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
            
            # Update sigma
            sigma *= np.exp(cs / 5 * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-12, 100)
            
            # Generate new population
            pop = [x + sigma * (B @ (D * np.random.standard_normal(dim))) for _ in range(lambda_)]
            pop_f = [evaluate(np.clip(p, -5, 5)) for p in pop]
            
            # Sort
            idx = np.argsort(pop_f)
            pop = [pop[i] for i in idx]
            pop_f = [pop_f[i] for i in idx]
        
        # Phase 2: (1+1)-ES for exploitation on the best point found
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # start with smaller sigma for local refinement
            
            while calls_made < budget and time.time() < deadline:
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                if sigma < 1e-10:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
