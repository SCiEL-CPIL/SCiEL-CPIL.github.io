"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 330s kill
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(n))  # population size ~ 11 for n=5
        # Actually let's use a standard population size
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        
        # For n=5: lambda = 4 + 3*1.609 = 4 + 4.8 = 8 (int)
        # Let's use a slightly larger population for better exploration
        lambda_ = max(lambda_, 8)
        
        # We have budget=500 evaluations. Each generation uses lambda_ evaluations.
        # Number of generations = (budget - 1) / lambda_ (first eval is initial)
        
        # Initial point
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        eval_count = 1

        # CMA-ES state
        sigma = 2.0
        mean = x.copy()
        C = np.eye(n)
        C_inv = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Learning rates
        mu = lambda_ // 2  # weights for top mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Precompute Cholesky decomposition of C
        try:
            C_chol = np.linalg.cholesky(C)
        except:
            C_chol = np.eye(n)
        
        # Inverse square root of C for ps update
        C_inv_sqrt = np.linalg.inv(C_chol)

        for gen in range(budget):
            # Check time
            if time.time() > deadline:
                break
            if eval_count >= budget:
                break
            
            # Generate lambda_ samples
            offspring = []
            offspring_fx = []
            
            for i in range(lambda_):
                # Check time and budget
                if time.time() > deadline or eval_count >= budget:
                    break
                    
                z = rng.standard_normal(n)
                y = mean + sigma * (C_chol @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                eval_count += 1
                offspring.append(y)
                offspring_fx.append(fy)
                
                # Update best
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            
            if len(offspring) == 0:
                break
            
            # Sort by fitness
            indices = np.argsort(offspring_fx)
            offspring = [offspring[i] for i in indices]
            offspring_fx = [offspring_fx[i] for i in indices]
            
            # Update mean
            mean_new = np.sum([weights[i] * offspring[i] for i in range(min(mu, len(offspring)))], axis=0)
            
            # Update evolution paths
            # Need to compute the weighted difference
            d = np.zeros(n)
            for i in range(min(mu, len(offspring))):
                d += weights[i] * (offspring[i] - mean)
            
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ d) / sigma
            
            # Update ps
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(2 * (gen + 1)))) < \
                   (1 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
            hsig = float(hsig)
            ps = (1 - c_sigma) * ps + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d / sigma
            
            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_sigma * (2 - c_sigma)
            rank_one = np.outer(ps, ps)
            rank_mu = np.zeros((n, n))
            for i in range(min(mu, len(offspring))):
                vi = (offspring[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(vi, vi)
            
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_hsig - 1) * np.eye(n) + c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                C_chol = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(C_chol)
            except:
                # Regularize
                C += 1e-8 * np.eye(n)
                try:
                    C_chol = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(C_chol)
                except:
                    C_chol = np.eye(n)
                    C_inv_sqrt = np.eye(n)
            
            # Update sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Adaptive sigma adjustment based on success rate
            # Simple: if we had good progress, increase sigma slightly
            if len(offspring_fx) > 0 and offspring_fx[0] < fx:
                sigma *= 1.1
            else:
                sigma *= 0.95
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            mean = mean_new
            fx = best_fx  # Track the best seen so far for sigma adaptation

        return best_x, best_fx
# EVOLVE-BLOCK-END
