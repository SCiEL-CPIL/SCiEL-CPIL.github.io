import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        lambda_pop = max(4 + int(3 * np.log(d)), 10)  # population size
        mu = lambda_pop // 2  # number of parents
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()  # effective population size
        
        # Initial parameters
        x_best = np.zeros(d)
        f_best = func(x_best)
        count = 1
        
        sigma = 2.5
        mean = x_best.copy()
        C = np.eye(d)  # covariance matrix
        C_sqrt_inv = np.eye(d)  # inverse square root of C
        ps = np.zeros(d)  # path for sigma
        pc = np.zeros(d)  # path for C
        
        # Adaptation parameters
        cs = (mu_eff + 2) / (d + mu_eff + 5)
        ds = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + cs
        c1 = (2 + 2 / (d + 1.3)) / ((d + 2) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        
        # Restart mechanism
        restart_threshold = 100
        no_improve_count = 0
        best_f_so_far = f_best
        
        while count < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            for i in range(lambda_pop):
                z = rng.standard_normal(d)
                y = mean + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                if time.time() > deadline:
                    break
                fy = func(y)
                offspring_f.append(fy)
                count += 1
            
            if len(offspring_f) < lambda_pop:
                break
            
            # Sort by fitness
            indices = np.argsort(offspring_f)
            offspring_sorted = [offspring[i] for i in indices]
            offspring_f_sorted = [offspring_f[i] for i in indices]
            
            # Update best
            if offspring_f_sorted[0] < f_best:
                f_best = offspring_f_sorted[0]
                x_best = offspring_sorted[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Check for restart
            if no_improve_count > restart_threshold and count < budget - 50:
                # Restart with smaller sigma
                sigma *= 0.5
                no_improve_count = 0
                continue
            
            # Update mean (recombination)
            new_mean = np.zeros(d)
            for i in range(mu):
                new_mean += weights[i] * offspring_sorted[i]
            mean = new_mean
            
            # Update sigma (1/5 rule)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt_inv @ (new_mean - mean) / sigma)
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (count - 1))) < (1.4 + 1 / d) * chi_n:
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            v = (new_mean - mean) / sigma
            pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mu_eff) * v
            rank_one = np.outer(v, v)
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                vi = (offspring_sorted[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(vi, vi)
            
            C = (1 + c1 * (cmu + 1) - c1 - cmu) * C + c1 * (rank_one + (pc @ pc - (1 - c1) * C)) + cmu * rank_mu
            C = (C + C.T) / 2  # symmetrize
            
            # Update C_sqrt_inv using eigenvalue decomposition (expensive but stable)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)  # ensure positive definite
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                # Fallback: reset C to identity if numerical issues
                C = np.eye(d)
                C_sqrt_inv = np.eye(d)
        
        return x_best, f_best
