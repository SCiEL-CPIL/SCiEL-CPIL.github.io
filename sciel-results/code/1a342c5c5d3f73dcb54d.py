import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        best_x = np.random.uniform(-5, 5, n)
        best_f = func(best_x)
        calls = 1
        
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        lambda_pop = 12
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        def run_cmaes(x0, sigma0, max_evals):
            nonlocal best_x, best_f, calls, deadline
            
            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            evals_used = 0
            stagnation_counter = 0
            last_best_f = best_f
            
            while evals_used < max_evals and calls < self.budget:
                if time.time() > deadline:
                    return
                
                # Generate population
                ys = []
                fs = []
                for i in range(lambda_pop):
                    if time.time() > deadline:
                        return
                    y = x + sigma * C_sqrt @ rng.standard_normal(n)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    evals_used += 1
                    ys.append(y)
                    fs.append(fy)
                    
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                        
                # Sort by fitness
                indices = np.argsort(fs)
                ys = [ys[i] for i in indices]
                fs = [fs[i] for i in indices]
                
                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * ys[i]
                
                # Update step size (sigma)
                y_sigma = (C_sqrt @ y_w) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals_used + 1) / (n + 1)))) / chi_n < 1.4 + 2 / (n + 1)
                sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update covariance matrix
                if not h_sigma:
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
                else:
                    p_c = (1 - c_c) * p_c
                    
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C
                for i in range(mu):
                    y_i = ys[i]
                    C += c_1 * np.outer(p_c, p_c) + c_mu * weights[i] * np.outer(y_i, y_i)
                
                # Re-decompose C
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-12, None)
                C_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
                
                # Update mean
                x = x + y_w
                
                # Check for stagnation or convergence
                if fs[0] - fs[-1] < 1e-12 or np.linalg.norm(y_w) < 1e-12:
                    break
                    
                if stagnation_counter > lambda_pop * 2:
                    break
                    
            return

        # Multi-start strategy
        # 5 restarts
        for i in range(5):
            if calls >= self.budget or time.time() > deadline:
                break
                
            if i < 3:
                # Random start
                current_start = rng.uniform(-5, 5, n)
                current_sigma = 2.0
            else:
                # Exploit best
                current_start = best_x + rng.normal(0, 0.5, n)
                current_sigma = 0.5
                
            run_cmaes(current_start, current_sigma, 100)
            
        return best_x, best_f
