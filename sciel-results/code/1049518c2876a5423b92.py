import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size: lambda = 4 + 3*ln(n)
        lambda_ = 4 + int(3 * np.log(n))
        # Number of selected points
        mu = lambda_ // 2
        
        # Weights for the selected points
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        
        # Effective population size
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0) - 1.0)) + cs
        
        # Initial mean and step size
        mean = np.zeros(n)
        sigma = 2.0  # Initial step size to explore [-5, 5]
        
        # Covariance matrix (initially identity)
        C = np.eye(n)
        
        # Eigen decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.sqrt(np.maximum(eigvals, 1e-16))
        
        # Cumulative step size
        ps = np.zeros(n)
        # Covariance adaptation
        pc = np.zeros(n)
        
        # Best so far
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Precompute weight square root for sampling
        sqrt_weights = np.sqrt(weights)
        
        while calls < budget:
            if time.time() - start_time > 280:
                break
            
            # Generate population
            population = np.zeros((lambda_, n))
            f_values = np.zeros(lambda_)
            
            for i in range(lambda_):
                # Sample from N(0, C)
                z = np.random.randn(n)
                y = eigvecs @ (eigvals * z)
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                population[i] = x
                f_values[i] = func(x)
                calls += 1
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            # Sort by function value
            order = np.argsort(f_values)
            population = population[order]
            f_values = f_values[order]
            
            # Update best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = population[0].copy()
            
            # Select top mu points
            selected = population[:mu]
            
            # New mean
            new_mean = np.sum(weights[:, np.newaxis] * selected, axis=0)
            
            # Step vector
            y = (new_mean - mean) / sigma
            
            # Cumulative step size adaptation
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * y
            
            # hsig: indicator if ps is within expected range
            hsig = np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2.0 / (calls - lambda_))) < (1.3 + 2.0 / (n + 1.0)) * np.sqrt(1.0 - (1.0 - cs) ** (2.0 / (calls - lambda_)))
            
            # Covariance matrix adaptation
            if hsig:
                pc = (1.0 - cs) * pc + cs * y
            
            # Rank-one and rank-mu updates
            delta_hs = (1.0 if hsig else 0.0) * cs * (2.0 - cs)
            C = (1.0 - cs * (c1 + cmu) - delta_hs) * C
            C += (1.0 + c1 * (1.0 / c1 - 1.0) - cs) * np.outer(pc, pc)
            
            # Rank-mu update
            for i in range(mu):
                zi = (selected[i] - mean) / sigma
                C += cmu * weights[i] * np.outer(zi, zi)
            
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-16)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            eigvals = np.sqrt(eigvals)
            
            # Step size adaptation
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2.0 / (calls - lambda_))) - 1.0))
            
            # Update mean
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            
            # Convergence check
            if sigma < 1e-10:
                break

        return best_x, best_f
