"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Standard population size: lambda = 4 + floor(3 * log(N))
        # For N=5, lambda = 10.
        # However, for a highly multimodal function with a limited budget (500),
        # a larger population helps with global exploration in the early stages.
        # Let's try lambda = 15 or 20 to cover the [-5, 5]^5 space better initially.
        # 500 / 20 = 25 generations. This is very few.
        # Maybe standard lambda=10 is better to allow more generations?
        # Let's stick to standard lambda=10 but with a large initial sigma.
        
        lambda_pop = 10
        mu = lambda_pop // 2  # Selection pressure
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initialization
        # The optimum is in [-4, 4]^5. Center at 0.
        # Initial sigma should be large enough to cover the domain.
        # Domain is [-5, 5]. Distance from center to corner is 5*sqrt(5) ~ 11.
        # Sigma = 2.0 seems reasonable to cover most of the space in a few steps.
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        # Covariance matrix and evolution paths
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim**2))
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Pre-generate the first population to ensure we use the budget efficiently
        # and handle the time limit safely.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            # We need to generate lambda_pop points.
            # If we run out of budget mid-generation, we just stop.
            # To be safe, let's check if we have enough budget for a full generation.
            # If not, we can generate partial or just stop.
            # Given lambda=10 and budget=500, we will have full generations until the end.
            # 500 // 10 = 50 generations exactly.
            
            if budget - calls_made < lambda_pop:
                # Not enough for a full generation. 
                # We could do a partial generation or just stop.
                # Stopping is safer to avoid complex partial updates.
                # Or we can reduce lambda for the last few steps?
                # Let's just stop if we can't do a full generation to keep CMA-ES logic clean.
                # Actually, we can just generate as many as we can and update with those?
                # No, CMA-ES update assumes fixed lambda.
                # Let's just break if we can't do a full generation.
                # Wait, 500 is divisible by 10. So this case might not happen unless time runs out.
                # If time runs out, we break anyway.
                pass 

            population = []
            fitness = []
            
            for i in range(lambda_pop):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Sample
                z = rng.standard_normal(dim)
                y = x_mean + sigma * z
                # Clip to box
                y = np.clip(y, -5.0, 5.0)
                
                # Evaluate
                fy = func(y)
                calls_made += 1
                budget -= 1
                
                population.append(y)
                fitness.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(population) < lambda_pop:
                break
            
            # Sort by fitness
            idx = np.argsort(fitness)
            sorted_pop = [population[i] for i in idx]
            sorted_fit = [fitness[i] for i in idx]
            
            # Update mean
            # x_mean = sum(weights[i] * sorted_pop[i]) for i in range(mu)
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * sorted_pop[i]
            
            # Update sigma
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C^{-1/2} * (new_mean - x_mean))
            # We need C^{-1/2}. For dim=5, we can use eigh.
            # But doing eigh every generation might be slow? 5x5 is very fast.
            
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            step = new_mean - x_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ step)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * step
            
            # Update covariance C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - c_c) * (1 - c_c + c_c)) * I + c_mu * sum(weights[i] * (y_i - x_mean) * (y_i - x_mean).T)
            
            # Correction term for pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1) / lambda_pop)) < (1 + 1.4 / dim) * chi_n)
            
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (pc * pc.T + (1 - hsig) * (1 - c_c) * (1 - c_c + c_c) * np.eye(dim))
            
            for i in range(mu):
                diff = sorted_pop[i] - x_mean
                C += c_mu * weights[i] * np.outer(diff, diff)
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            x_mean = new_mean

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            x = np.zeros(dim)
            fx = func(x)
            return x, fx

        return best_x, best_f
# EVOLVE-BLOCK-END
