# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with multi-start strategy for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin
        
        N = self.dim
        lambda_pop = 4 + int(3 * np.log(N))
        
        # Strategy: Multi-start CMA-ES with decreasing budgets
        # This balances exploration (many restarts) and exploitation (longer first runs)
        restart_budgets = [100, 80, 60, 40, 20]
        
        # Adjust if total budget is different
        total_restart_budget = sum(restart_budgets)
        if total_restart_budget > self.budget:
            scale = self.budget / total_restart_budget
            restart_budgets = [max(10, int(b * scale)) for b in restart_budgets]
        elif total_restart_budget < self.budget:
            # Add remaining to the first restart or distribute
            # For simplicity, just use the defined budgets, ignoring the rest or adding to first
            # Let's add the rest to the first restart
            restart_budgets[0] += (self.budget - total_restart_budget)

        best_x = None
        best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        for budget_r in restart_budgets:
            if time.time() > deadline:
                break
                
            if budget_r < lambda_pop:
                # Not enough budget for one generation, just sample a few points
                # and pick the best
                for _ in range(budget_r):
                    x = rng.uniform(-5, 5, N)
                    fx = func(x)
                    if fx < best_f:
                        best_f = fx
                        best_x = x
                continue

            # Initialize CMA-ES
            mean = rng.uniform(-5, 5, N)
            sigma = 2.0
            C = np.eye(N)
            pc = np.zeros(N)
            cs = 0.0
            
            # Parameters
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights**2)
            
            cmu = 2 / ((N + 4) ** 2 + 10)
            cc = (4 + N / 4) / ((N + 2) ** 2 + 4)
            cs_ = (2 + cmu) / (N + 2 + cmu)
            damps = 1 + 2 * max(0, np.sqrt((N - 1) / (N + 3) - 1)) + cs_
            chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N**2))
            
            max_generations = budget_r // lambda_pop
            
            for gen in range(max_generations):
                if time.time() > deadline:
                    break
                    
                # Sample
                try:
                    B, d = np.linalg.eigh(C)
                    D = 1 / np.sqrt(np.maximum(d, 1e-12))
                except:
                    break
                    
                Z = rng.standard_normal((lambda_pop, N))
                Y = Z @ (B * D)
                X = mean[None, :] + sigma * Y
                
                F = np.array([func(x) for x in X])
                
                idx = np.argsort(F)
                X_sorted = X[idx]
                F_sorted = F[idx]
                
                if F_sorted[0] < best_f:
                    best_f = F_sorted[0]
                    best_x = X_sorted[0].copy()
                    
                # Update
                new_mean = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
                y_mean = (new_mean - mean) / sigma
                pc = (1 - cs_) * pc + cs_ * y_mean
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs_)**(2 * (gen + 1)))) / chiN < 1.4 + 2 / (N + 1)
                
                if hsig:
                    cs = (1 - cs_) * cs + cs_ * (np.linalg.norm(pc) / chiN - 1)
                    C += (1 + (1 - hsig) * cmu * mueff - 1) * cc * np.outer(pc, pc)
                else:
                    cs = (1 - cs_) * cs
                    
                sigma *= np.exp(cs / damps * (np.linalg.norm(pc) / chiN - 1))
                
                v = X_sorted[:mu] - mean
                C += cmu * np.sum(weights[:, None, None] * v[:, :, None] * v[:, None, :], axis=0)
                
                C = (C + C.T) / 2
                C += 1e-12 * np.eye(N)
                
                mean = new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
