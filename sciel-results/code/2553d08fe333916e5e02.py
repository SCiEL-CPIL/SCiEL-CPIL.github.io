"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Adaptive Hyperparameter Annealing (AHA) with covariance adaptation.
    
    Uses a population of 12 candidates with adaptive step sizes and
    full covariance matrix adaptation based on success rates.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Population size
        pop_size = 12
        
        rng = np.random.default_rng(42)
        
        # Initialize population with diverse starting points
        x_best = rng.uniform(-4, 4, dim)
        f_best = func(x_best)
        calls_made = 1
        
        # Create initial population
        population = [x_best.copy()]
        f_values = [f_best]
        
        for i in range(1, pop_size):
            x_new = rng.uniform(-5, 5, dim)
            f_new = func(x_new)
            population.append(x_new)
            f_values.append(f_new)
            calls_made += 1
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()
        
        # Sort initial population
        indices = np.argsort(f_values)
        population = [population[i] for i in indices]
        f_values = [f_values[i] for i in indices]
        
        # CMA-ES parameters
        mu = pop_size // 2  # number of parents
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size
        sigma = 1.0
        min_sigma = 1e-8
        max_sigma = 10.0
        
        # Covariance matrix
        C = np.eye(dim)
        Csqr = np.eye(dim)  # C^(1/2)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff) / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Damping for covariance update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Track improvement for adaptive annealing
        recent_improvements = []
        max_improvement_window = 30
        
        # Main optimization loop
        remaining_budget = budget - calls_made
        
        for gen in range(remaining_budget):
            if calls_made >= budget:
                break
            
            # Compute mean from weighted recombination of top mu
            m = np.zeros(dim)
            for i in range(mu):
                m += weights[i] * population[i]
            
            # Generate offspring
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                x_new = m + sigma * Csqr @ z
                x_new = np.clip(x_new, -5, 5)
                offspring.append(x_new)
            
            # Evaluate offspring
            offspring_f = []
            improved_this_gen = False
            for x in offspring:
                if calls_made >= budget:
                    break
                f_val = func(x)
                offspring_f.append(f_val)
                calls_made += 1
                
                if f_val < f_best:
                    f_best = f_val
                    x_best = x.copy()
                    improved_this_gen = True
            
            # If we didn't evaluate all offspring, pad with best
            while len(offspring_f) < pop_size:
                offspring_f.append(f_best)
            
            # Combine parent and offspring
            combined = population + offspring
            combined_f = f_values + offspring_f
            
            # Track improvement
            recent_improvements.append(improved_this_gen)
            if len(recent_improvements) > max_improvement_window:
                recent_improvements = recent_improvements[-max_improvement_window:]
            
            # Select best pop_size individuals
            indices = np.argsort(combined_f)
            population = [combined[i] for i in indices[:pop_size]]
            f_values = [combined_f[i] for i in indices[:pop_size]]
            
            # Update CMA-ES parameters using the selected population
            # Compute mean again from selected population
            m_new = np.zeros(dim)
            for i in range(mu):
                m_new += weights[i] * population[i]
            
            # Compute y vectors
            y_w = np.zeros(dim)
            y_sigma = np.zeros(dim)
            for i in range(mu):
                y_i = population[i] - m
                y_w += weights[i] * y_i
                y_sigma += weights[i] * y_i / sigma
            
            # Update sigma
            norm_y_sigma = np.linalg.norm(y_sigma)
            h_sigma = float(norm_y_sigma / chi_n > 1.4 + 2 / (dim + 1))
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sigma * y_sigma / chi_n
            
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * 1)) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma * (norm_y_sigma / chi_n - 1))
            
            sigma = np.clip(sigma, min_sigma, max_sigma)
            
            # Update covariance matrix
            y_w_normalized = y_w / sigma
            p_c = (1 - c_c) * p_c + c_c * h_sigma * y_w_normalized
            
            C = (1 + c_1 * c_mu * mu_eff - c_1) * C + c_1 * np.outer(p_c, p_c)
            
            # Update rank-mu correction
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (population[i] - m) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C += c_mu * rank_mu
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update Csqr
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            Csqr = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        return x_best, f_best
# EVOLVE-BLOCK-END
