import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        # For dim=5, a population size of 8 is reasonable.
        # lambda = dim + int(3 * log(dim)) = 5 + 6 = 11? Or smaller for budget?
        # Let's use lambda = 6 or 8.
        # Given 500 budget, we can afford ~60 generations if lambda=8.
        # Let's use lambda = 8.
        
        pop_size = 8
        # Initial sigma
        # Domain is [-5, 5], width 10. 
        # Standard deviation of 1.0 is a good start.
        
        # We will run multiple instances of CMA-ES.
        # How many? 
        # If one CMA-ES run takes ~100-150 calls to converge or stagnate, we can do 3-4 runs.
        # Let's set a max_calls_per_run limit to force restarts.
        
        max_calls_per_run = 120
        
        # Simple CMA-ES implementation
        def cmaes(x0, sigma0, max_calls):
            nonlocal calls_made, best_x, best_f
            
            n = dim
            # Initialize
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            p_c = np.zeros(n)
            p_s = np.zeros(n)
            
            # Parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_c = min(c_c, 1.0)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 1 + weights[1]) / ((n + 2)**2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
            
            # Initial population
            # We don't evaluate x0 yet in the loop, we evaluate the sample.
            # But we need to track calls.
            
            local_calls = 0
            local_best_x = x0.copy()
            local_best_f = float('inf')
            
            # We will generate samples and evaluate them.
            # To be efficient, we evaluate the whole population at once.
            
            while local_calls < max_calls and can_call():
                # Generate samples
                # B = L^{-1}
                # We need to update C and B each generation.
                
                # Sample
                # z ~ N(0, I)
                # x = x_mean + sigma * B @ z
                
                # For the first iteration, C=I, B=I.
                
                # We evaluate pop_size points.
                # If we run out of budget mid-population, we stop.
                
                # Generate z
                z = np.random.randn(pop_size, n)
                # Transform
                # B is the inverse square root of C.
                # We need to maintain B.
                # Let's use a simplified update or just recompute B from C each time?
                # Recomputing B (eigh) is O(n^3) which is fine for n=5.
                
                # Update B
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                # Wait, B @ B.T = C^{-1}? No, B should be such that B @ B.T = C.
                # So B = eigvecs @ sqrt(eigvals) @ eigvecs.T
                B = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                
                samples = x_mean[None, :] + sigma * (z @ B.T)
                
                # Evaluate
                f_values = np.zeros(pop_size)
                for i in range(pop_size):
                    if not can_call():
                        break
                    f_values[i] = evaluate(samples[i])
                    local_calls += 1
                    if f_values[i] < local_best_f:
                        local_best_f = f_values[i]
                        local_best_x = samples[i].copy()
                
                if local_calls >= max_calls or not can_call():
                    break
                    
                # Sort
                order = np.argsort(f_values)
                x_selected = samples[order]
                f_selected = f_values[order]
                
                # Update mean
                # x_mean_new = sum(weights[i] * x_selected[i]) for i in range(mu)
                x_mean_new = np.sum(weights[:mu, None] * x_selected[:mu], axis=0)
                
                # Update step size control variable
                # p_s = (1 - c_sigma) * p_s + c_sigma * sqrt(mu_eff) * B^{-1} * (x_mean_new - x_mean) / sigma
                # B^{-1} = eigvecs @ inv(sqrt(eigvals)) @ eigvecs.T
                B_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                y_mean = (x_mean_new - x_mean) / sigma
                p_s = (1 - c_sigma) * p_s + c_sigma * np.sqrt(mu_eff) * (B_inv @ y_mean)
                
                # Update C
                # h_sigma = ||p_s|| / (1 - (1 - c_sigma)^(gen+1)) / chi_n
                # We don't track gen count easily in this loop without a counter, 
                # but we can approximate or just use a simple heuristic.
                # Let's use a standard heuristic for h_sigma.
                # Since we don't have the history of p_s, we approximate h_sigma = 1.0 if ||p_s|| < 1.75 * chi_n else 0.0?
                # Actually, the standard condition is:
                # h_sigma = 1 if ||p_s|| / (1 - (1-c_sigma)^(g+1)) < 1.4 + 2/(n+1) else 0
                # Without g, we can't be precise. Let's assume h_sigma = 1.0 for simplicity,
                # or use a fixed threshold.
                # For n=5, 1.4 + 2/6 = 1.73.
                # Let's just check if ||p_s|| is small.
                norm_ps = np.linalg.norm(p_s)
                # chi_n is approx sqrt(5) * (1 - 1/20 + ...) ~ 2.2
                # If norm_ps < 1.75 * chi_n, h_sigma = 1
                h_sigma = 1.0 if norm_ps < 1.75 * chi_n else 0.0
                
                # Delta term
                delta_term = (c_1 + c_mu) * (1 - (1 - c_1 - c_mu)**(2 * local_calls)) # Approximation
                # Actually, the formula is:
                # C = (1 - c_1 - c_mu + c_mu * h_sigma * delta_term) * C + c_1 * p_c @ p_c.T + c_mu * sum(weights[i] * y_i @ y_i.T)
                
                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * c_c * y_mean
                
                # Compute y_i
                y_i = (x_selected[:mu] - x_mean) / sigma
                # Sum of outer products
                sum_outer = np.zeros((n, n))
                for i in range(mu):
                    sum_outer += weights[i] * np.outer(y_i[i], y_i[i])
                
                # Update C
                # Coefficient for C
                c_C = 1 - c_1 - c_mu + c_mu * h_sigma * delta_term
                C = c_C * C + c_1 * np.outer(p_c, p_c) + c_mu * sum_outer
                
                # Update sigma
                # sigma = sigma * exp(c_sigma / chi_n * (||p_s|| - chi_n))
                sigma *= np.exp(c_sigma / chi_n * (norm_ps - chi_n))
                
                # Ensure sigma is within bounds
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Update mean
                x_mean = x_mean_new
                
                # Check for convergence
                if np.linalg.norm(x_mean_new - x_mean) < 1e-6 and sigma < 1e-4:
                    break
                    
                # If sigma is too small, break
                if sigma < 1e-6:
                    break

            return local_best_x, local_best_f

        # Multi-start CMA-ES
        # We'll start from a few random points or the origin.
        # Given the shift is hidden, starting from 0 is a good bet.
        # We'll run CMA-ES, and if it stops early or stagnates, we restart.
        
        # To ensure diversity, we can jitter the starting points for restarts.
        
        num_restarts = 4
        for r in range(num_restarts):
            if not can_call():
                break
            
            # Start point
            if r == 0:
                x_start = np.zeros(dim)
            else:
                # Random start within [-4, 4]
                x_start = np.random.uniform(-4, 4, dim)
                
            # Initial sigma
            sigma_start = 1.0
            
            # Run CMA-ES
            local_best_x, local_best_f = cmaes(x_start, sigma_start, max_calls_per_run)
            
            # Update global best
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()
            
            # If we found a very good solution, we might stop early?
            # Or just let the budget run out.
            
            # If the run used very few calls (converged fast), we can do another restart.
            # The loop handles this via can_call().

        return best_x
