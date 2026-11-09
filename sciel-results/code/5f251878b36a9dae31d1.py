"""CMA-ES with IPOP restart strategy for BBOB f7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with restart strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(n))  # Population size
        mu = lambda_ // 2                  # Number of weights
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Restart strategy parameters
        max_generations_no_improve = 20 * n
        sigma_reduction_factor = 0.5
        min_sigma = 1e-6
        
        # Initial state
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-5, 5, n)
        sigma = 5.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Track progress for restart
        gen_count = 0
        no_improve_count = 0
        prev_best_f = best_f
        
        while calls_made < budget and time.time() < deadline:
            # Generate offspring
            # Calculate step size
            hsig = (np.sqrt((1 + 2 / (cs + 1) + cs) * (1 - cs)) - 1) * (1 + cc)
            
            # Sample
            # Ensure C is symmetric and positive definite (it should be)
            # Use Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes non-PD due to numerical issues, reset to identity
                C = np.eye(n)
                L = np.eye(n)
            
            # Generate standard normal samples
            z = rng.standard_normal((lambda_, n))
            # Transform to search space
            y = np.dot(z, L.T)
            x_offspring = np.array([x_mean + sigma * y[i] for i in range(lambda_)])
            
            # Clip to box
            x_offspring = np.clip(x_offspring, -5, 5)
            
            # Evaluate offspring
            f_offspring = np.array([func(x) for x in x_offspring])
            calls_made += lambda_
            
            # Sort by fitness
            idx = np.argsort(f_offspring)
            x_offspring = x_offspring[idx]
            f_offspring = f_offspring[idx]
            
            # Update best
            if f_offspring[0] < best_f:
                best_f = f_offspring[0]
                best_x = x_offspring[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Check for restart
            if no_improve_count > max_generations_no_improve:
                # Restart at best point with smaller sigma
                if sigma * sigma_reduction_factor < min_sigma:
                    break
                x_mean = best_x.copy()
                sigma *= sigma_reduction_factor
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                no_improve_count = 0
                continue
            
            # Update mean
            x_new_mean = np.sum([weights[i] * x_offspring[i] for i in range(mu)], axis=0)
            
            # Update covariance matrix adaptation
            # hsig calculation for ps update
            # Actually, hsig is used to decide whether to update pc with the direction of the step
            # Standard CMA-ES update:
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / np.linalg.norm(np.dot(L, x_new_mean - x_mean)) * (x_new_mean - x_mean)
            
            # Update pc
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen_count + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            # Rank-1 and Rank-mu update of C
            v = (x_new_mean - x_mean) / sigma
            C = (1 + c1 * hsig * (cc * (2 - cc)) - c1 - cmu) * C \
                + c1 * np.outer(pc, pc) \
                + cmu * np.sum([weights[i] * np.outer(v - (x_offspring[i] - x_mean) / sigma, v - (x_offspring[i] - x_mean) / sigma) for i in range(mu)], axis=0)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            x_mean = x_new_mean
            
            gen_count += 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
