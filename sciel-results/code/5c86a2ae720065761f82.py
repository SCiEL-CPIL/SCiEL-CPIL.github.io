# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Adaptive Hyperparameter Annealing (AHA) with a single population of 10 candidates.
    
    This approach uses a population of 10 candidates. The mutation step size (sigma)
    and covariance adaptation rates are annealed based on the current improvement
    rate (success rate) rather than just time. This allows the search to dynamically
    adjust its exploration vs. exploitation balance.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Population size as specified in directive
        pop_size = 10
        
        rng = np.random.default_rng(42)
        
        # Initialize population with diverse starting points
        # Use a mix of random points and a center point to ensure coverage
        x_best = rng.uniform(-4, 4, dim)
        f_best = func(x_best)
        calls_made = 1
        
        # Create initial population
        population = [x_best.copy()]
        f_values = [f_best]
        
        for i in range(1, pop_size):
            # Mix of random and perturbed best
            if i % 2 == 0:
                x_new = rng.uniform(-5, 5, dim)
            else:
                x_new = x_best + rng.standard_normal(dim) * 1.0
                x_new = np.clip(x_new, -5, 5)
            
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
        mu = pop_size // 2  # number of parents (5)
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
        
        # Base learning rates
        c_sigma_base = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c_base = (4 - mu_eff) / (dim + 4)
        c_1_base = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu_base = min(1 - c_1_base, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Damping for covariance update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # AHA Parameters
        # We track success rate to anneal sigma and covariance rates
        success_history = []
        max_history = 20
        
        # Main optimization loop
        remaining_budget = budget - calls_made
        gen = 0
        
        while calls_made < budget:
            gen += 1
            
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
            p_sigma = (1 - c_sigma_base) * p_sigma + c_sigma_base * h_sigma * y_sigma / chi_n
            
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma_base) ** (2 * 1)) < (1.4 + 2 / (dim + 1)) * chi_n:
                # Standard CMA-ES sigma update
                sigma *= np.exp(c_sigma_base * (norm_y_sigma / chi_n - 1))
            
            # AHA: Adaptive Sigma Annealing based on Success Rate
            # If we are improving frequently, we might be moving too slowly or in the wrong direction?
            # Actually, in AHA, if success rate is high, we can afford to increase step size to explore more.
            # If success rate is low, we should decrease step size to exploit.
            # However, standard CMA-ES already adapts sigma. The directive says "annealed based on current improvement rate".
            # Let's add an additional modulation factor.
            
            success_history.append(1.0 if improved_this_gen else 0.0)
            if len(success_history) > max_history:
                success_history = success_history[-max_history:]
            
            success_rate = np.mean(success_history)
            
            # Modulate sigma based on success rate
            # If success rate is very high (> 0.5), we might be stuck in a valley or moving well.
            # If success rate is 0, we are stuck.
            # Let's use a gentle annealing:
            # If success_rate is low, decrease sigma more aggressively.
            # If success_rate is high, allow sigma to be slightly larger.
            modulation = 1.0 + 0.2 * (success_rate - 0.5)
            sigma *= modulation
            
            sigma = np.clip(sigma, min_sigma, max_sigma)
            
            # AHA: Adaptive Covariance Rates
            # As success rate drops, we should slow down covariance adaptation to stabilize.
            # As success rate is high, we can adapt faster.
            # We anneal c_c and c_1.
            
            # Base rates are c_c_base and c_1_base.
            # Let's scale them by a factor that depends on success rate and generation.
            # Early on, we want fast adaptation. Later, if stuck, slow down.
            
            # A simple heuristic: 
            # If success_rate is low, reduce adaptation rates to prevent oscillation.
            # If success_rate is high, keep rates high.
            
            # We can use an exponential decay of the "aggressiveness" based on lack of improvement.
            # Or simply link to success_rate directly.
            
            # Let's try: 
            # c_c_eff = c_c_base * (0.5 + 0.5 * success_rate)
            # c_1_eff = c_1_base * (0.5 + 0.5 * success_rate)
            # c_mu_eff = c_mu_base * (0.5 + 0.5 * success_rate)
            
            # This effectively anneals the covariance adaptation rates based on improvement.
            
            c_c_eff = c_c_base * (0.5 + 0.5 * success_rate)
            c_1_eff = c_1_base * (0.5 + 0.5 * success_rate)
            c_mu_eff = c_mu_base * (0.5 + 0.5 * success_rate)
            
            # Update covariance matrix
            y_w_normalized = y_w / sigma
            p_c = (1 - c_c_eff) * p_c + c_c_eff * h_sigma * y_w_normalized
            
            C = (1 + c_1_eff * c_mu_eff * mu_eff - c_1_eff) * C + c_1_eff * np.outer(p_c, p_c)
            
            # Update rank-mu correction
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (population[i] - m) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C += c_mu_eff * rank_mu
            
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
