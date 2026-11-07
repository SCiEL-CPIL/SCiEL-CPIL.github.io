import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0
        
        best_x_global = np.zeros(n)
        best_f_global = func(best_x_global)
        calls = 1
        
        # CMA-ES parameters
        lam = 8
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        def run_cmaes(x0, sigma0, max_calls):
            nonlocal calls, best_x_global, best_f_global
            
            m = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            hs = (1.4 + 2.0 / (n + 1)) * np.sqrt((mu_eff - 1) + 1)
            
            chs = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) - 1
            hs_threshold = (1 + 2 * max(0, (mu_eff - 1) / (n + 1))) * (1 - c_sigma)
            
            local_best_f = func(m)
            local_best_x = m.copy()
            if local_best_f < best_f_global:
                best_f_global = local_best_f
                best_x_global = local_best_x.copy()
            calls += 1
            
            no_improve = 0
            threshold = 1e-8
            
            while calls < budget and max_calls > 0:
                if time.time() > deadline:
                    break
                
                if budget - calls < lam:
                    break
                
                # Sample
                Z = np.random.standard_normal((lam, n))
                X = m + sigma * (Z @ (B * D).T)
                
                # Evaluate
                F = np.empty(lam)
                for i in range(lam):
                    F[i] = func(X[i])
                    calls += 1
                    max_calls -= 1
                
                # Sort
                indices = np.argsort(F)
                F = F[indices]
                X = X[indices]
                
                # Update best
                if F[0] < local_best_f:
                    local_best_f = F[0]
                    local_best_x = X[0].copy()
                    no_improve = 0
                    if local_best_f < best_f_global:
                        best_f_global = local_best_f
                        best_x_global = local_best_x.copy()
                else:
                    no_improve += 1
                
                # Check stagnation
                if no_improve > 20:
                    break
                
                # Recombination
                m_old = m
                m = np.sum([w[i] * X[i] for i in range(mu)], axis=0)
                
                # Step size adaptation
                y = (m - m_old) / sigma
                z = (B.T @ y) / D
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
                
                if np.linalg.norm(p_sigma) / hs < hs_threshold:
                    sigma *= np.exp(c_sigma / chs * (np.linalg.norm(p_sigma) / hs - 1))
                
                # Covariance matrix update
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
                
                sum_outer = np.zeros((n, n))
                for i in range(mu):
                    y_i = (X[i] - m_old) / sigma
                    sum_outer += w[i] * np.outer(y_i, y_i)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * np.eye(n)) + c_mu * sum_outer
                
                # Re-orthonormalize
                C = (C + C.T) / 2
                eigenvalues, B = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-12, None)
                D = np.sqrt(eigenvalues)
                
                if sigma < 1e-12:
                    break
            
            return local_best_x, local_best_f
        
        # First run
        x0, f0 = run_cmaes(np.zeros(n), 2.0, budget - calls)
        
        # If stagnated, try restart from best with larger sigma
        while calls < budget and time.time() < deadline:
            # Restart with larger step size
            sigma_restart = max(3.0, 2.0 * 2)
            x_restart, f_restart = run_cmaes(best_x_global.copy(), sigma_restart, budget - calls)
            
            if f_restart >= best_f_global - 1e-10:
                break
        
        return best_x_global, best_f_global
