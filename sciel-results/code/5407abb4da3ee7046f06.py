import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 280  # Safety margin
            
            # CMA-ES Parameters
            n = self.dim
            pop_size = max(4, int(4 + 3 * np.log(n)))  # ~11 for n=5
            sigma = 3.0
            mean = np.zeros(n)
            
            # CMA-ES State
            C = np.eye(n)
            alpha = 1.0 # Learning rate for covariance
            # Standard CMA-ES adaptation rates
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_cov = (2 + 2 * min(1, (mu_eff - 1) / (n + 1)) - c_sigma - 1 / c_sigma) / ((n + 1.3) * (n + 1 + 2 * mu_eff / n))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            
            # Normalization for step size adaptation
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            
            best_x = mean.copy()
            best_f = func(best_x)
            evals = 1
            
            # Track history for early stopping or restart?
            # For simplicity, run one long CMA-ES.
            
            while evals < self.budget:
                # Time check
                if time.time() > deadline:
                    break
                    
                # Generate population
                # We can only generate up to remaining budget
                current_pop_size = min(pop_size, self.budget - evals)
                if current_pop_size <= 0:
                    break
                
                # Sample
                # Use Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                    
                z = np.random.randn(current_pop_size, n)
                y = z @ L.T * sigma
                x_pop = mean + y
                x_pop = np.clip(x_pop, -5, 5)
                
                # Evaluate
                f_pop = np.array([func(x) for x in x_pop])
                evals += current_pop_size
                
                # Update best
                idx_best = np.argmin(f_pop)
                if f_pop[idx_best] < best_f:
                    best_f = f_pop[idx_best]
                    best_x = x_pop[idx_best].copy()
                
                # Sort
                idx_sorted = np.argsort(f_pop)
                x_sorted = x_pop[idx_sorted]
                f_sorted = f_pop[idx_sorted]
                
                # Update mean
                # Use only top mu
                m_eff = min(mu, current_pop_size)
                if m_eff == 0: continue
                w = weights[:m_eff]
                w = w / np.sum(w) # Renormalize if pop size changed
                mean_new = np.sum(w[:, None] * x_sorted[:m_eff], axis=0)
                
                # Update sigma (step size)
                # p_sigma update
                # Need p_sigma vector
                # ... (Implementing full CMA-ES is verbose. Let's use a simpler robust optimizer if CMA-ES is too risky to code correctly in one go).
