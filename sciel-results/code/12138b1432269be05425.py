"""Rewritten optimizer using CMA-ES with time budget guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with explicit time deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit, use 325s wall-clock kill margin
        
        rng = np.random.default_rng(42)
        
        # Initialize population mean at center of box
        mean = np.zeros(self.dim)
        # Initial sigma: box is [-5,5], so range 10. Use ~1/5 of range
        sigma = 1.5
        
        # Population size
        lambda_ = 4 + int(3 * np.log(self.dim))  # lambda = 4 + 3*log(d)
        mu = lambda_ // 2  # number of parents
        
        # Weights: log-linear weights, normalized
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        c1 = (2 * (mueff + 0.5)) / ((self.dim + 2)**2 + mueff + 5)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((self.dim + 2)**2 + mueff + 5))
        cs = (mueff + 1.7) / (self.dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + cs
        c_sigma = cs * (ds + 1)
        chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        # Covariance matrix
        C = np.eye(self.dim)
        # Eigenvalue decomposition for efficient update
        eigenvalues = np.ones(self.dim)
        eigenvectors = np.eye(self.dim)
        
        # State variables
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        best_x = mean.copy()
        best_f = np.inf
        num_calls = 0
        
        def check_time():
            return time.time() > deadline - 5  # 5s safety margin
        
        def evaluate(x):
            nonlocal num_calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            num_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Initial population evaluation with proper covariance transform
        for i in range(lambda_):
            if num_calls >= self.budget or check_time():
                break
            z = rng.standard_normal(self.dim)
            x_i = mean + sigma * (eigenvectors @ (np.sqrt(eigenvalues) * z))
            f_i = evaluate(x_i)
        
        while num_calls < self.budget and not check_time():
            # Generate offspring
            offspring = []
            fitness = []
            
            for i in range(lambda_):
                if num_calls >= self.budget or check_time():
                    break
                z = rng.standard_normal(self.dim)
                x_i = mean + sigma * (eigenvectors @ (np.sqrt(eigenvalues) * z))
                f_i = evaluate(x_i)
                offspring.append(x_i)
                fitness.append(f_i)
            
            if len(offspring) == 0:
                break
            
            # Sort by fitness
            sorted_indices = np.argsort(fitness)
            offspring = [offspring[i] for i in sorted_indices]
            fitness = [fitness[i] for i in sorted_indices]
            
            # Save old mean for path updates
            old_mean = mean.copy()
            
            # Update mean
            new_mean = np.zeros(self.dim)
            for i in range(min(mu, len(offspring))):
                new_mean += weights[i] * offspring[i]
            
            # Update path for covariance (uses old_mean)
            h_delta = 1 if np.linalg.norm(p_c) / np.sqrt(1 - (1-c1)**2) < (1.4 + 2/(self.dim+1)) * chiN else 0
            p_c = (1 - c1) * p_c + h_delta * np.sqrt(c1 * (2 - c1) * mueff) * (new_mean - old_mean) / sigma
            
            # Update mean
            mean = new_mean
            
            # Update covariance matrix using differences from OLD mean
            c1_term = c1 * (np.outer(p_c, p_c) - C)
            cmu_term = np.zeros((self.dim, self.dim))
            for i in range(min(mu, len(offspring))):
                z_i = eigenvectors.T @ (offspring[i] - old_mean) / sigma
                cmu_term += weights[i] * np.outer(z_i, z_i)
            cmu_term = cmu * cmu_term
            
            C = (1 - c1 - cmu) * C + c1_term + cmu_term
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            
            # Update step size control
            mean_shift = (new_mean - old_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (np.linalg.norm(mean_shift) - chiN)
            
            # Simple sigma adaptation based on success rate
            n_success = sum(1 for i in range(min(mu, len(offspring))) if fitness[i] < fitness[min(mu, len(offspring))-1])
            success_rate = n_success / max(1, min(mu, len(offspring)))
            if success_rate > 0.5:
                sigma *= 1.05
            else:
                sigma *= 0.95
            
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
