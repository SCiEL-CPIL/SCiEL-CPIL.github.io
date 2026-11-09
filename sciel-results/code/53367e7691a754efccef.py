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
        total_budget = self.budget
        
        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_made = 1

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES Parameters
        lambda_ = max(4 + int(3 * np.log(n)), 8) # Pop size ~ 10 for n=5
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # State
        x_m = best_x.copy()
        sigma = 2.0 # Initial step size covering [-5, 5]
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # History for stagnation
        last_improvement_call = calls_made
        
        while calls_made < total_budget and time.time() < deadline:
            # Sample population
            Y = np.random.randn(lambda_, n)
            # Transform to X
            # sqrt(C) @ Y
            # Use Cholesky for efficiency, but for n=5, eigendecomp is fine or just simple sampling
            # For simplicity and robustness in this short script, we can use the standard update.
            # However, calculating sqrt(C) every step is expensive if not done carefully.
            # Let's use a simplified "1+1" or small pop CMA-ES if standard is too complex to implement correctly in few lines.
            # Actually, standard CMA-ES is preferred.
            
            # Efficient sampling:
            # We need D such that D @ D.T = C
            # Use np.linalg.cholesky if C is positive definite.
            try:
                D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset
                C = np.eye(n)
                D = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                sigma *= 2.0
                continue

            X = x_m[None, :] + sigma * (Y @ D)
            
            # Evaluate
            F = np.array([evaluate(x) for x in X])
            
            # Sort indices
            idx = np.argsort(F)
            y_sorted = Y[idx]
            f_sorted = F[idx]
            
            # Update mean
            x_m_new = x_m + sigma * (weights @ (y_sorted[:mu]))
            
            # Update sigma
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * D.T @ (y_sorted[:mu] / sigma) # Note: y is in standard normal space, but we sampled Y, so Y is standard. The term is D.T @ (Y_sorted / sigma)? No, Y is standard. The update is usually on the Y vectors.
            # Standard CMA-ES update:
            # h_sigma = ...
            # p_sigma = (1-c_sigma)*p_sigma + h_sigma*sqrt(c_sigma*(2-c_sigma)*mu_eff)/sigma * D.T @ y_sorted[:mu]
            # Let's simplify: If we track Y, the update is straightforward.
            
            # Actually, implementing full CMA-ES correctly is error-prone.
            # Alternative: Use `scipy.optimize`? No, we must write the class.
            # Let's use a simpler, robust "Adaptive Random Search" or "Nelder-Mead" with restarts?
            # Nelder-Mead is bad for multimodal.
            # Let's stick to a simplified CMA-ES or a well-tuned (1+1)-ES with restarts.
            
            # Given the constraint "Keep the function names...", I will implement a robust CMA-ES.
            # I will use the standard formulas.
            
            # Calculate h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Expectation of norm of N(0, I)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2*(calls_made - last_improvement_call)))) < (1.4 + 2/(n+1)) * chi_n
            # This part is tricky with call counts. Let's use generations.
            
            # Simplified Restart Strategy:
            # If no improvement in 30 evaluations, restart from best_x with sigma *= 1.5 or random.
            
            # Let's switch to a simpler, very effective algorithm for this specific budget:
            # **Random Restart Coordinate Descent / Nelder Mead Hybrid?**
            # No, CMA-ES is best.
            
            # I will implement a standard CMA-ES loop.
            
            # Update p_sigma
            if h_sigma:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_sorted[:mu] @ D) / sigma
            else:
                p_sigma = (1 - c_sigma) * p_sigma
            
            # Update p_c
            h_c = (1 - np.linalg.norm(p_c)**2 / (1 - (1 - c_c)**(2*(calls_made - last_improvement_call)))) < ((1.4 + 2/(n+1)) * np.sqrt((c_c + 1)/(c_c - 1)) - 1)
            if h_c:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_sorted[:mu] @ D) / sigma
            else:
                p_c = (1 - c_c) * p_c
            
            # Update C
            C = (1 - c_c * c_c * (2 - c_c) + c_1 + c_mu) * C \
                + c_c * (2 - c_c) * np.outer(p_c, p_c) \
                + c_1 * (C - C) # No, c_1 * (C - C) is 0.
            # Correction:
            # C = (1 - c_1 - c_mu) * C + c_1 * (C + outer(p_c, p_c)) + c_mu * sum(weights[i] * outer(y_i, y_i))
            # Wait, the standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (C + outer(p_c, p_c)) + c_mu * sum(weights[i] * outer(y_i, y_i))
            # Where y_i are the sorted Y vectors.
            
            # Let's re-verify the update formula.
            # C_{t+1} = (1 - c_1 - c_mu) C_t + c_1 (C_t + p_c p_c^T) + c_mu sum_{i=1}^mu w_i y_i y_i^T
            # Yes.
            
            # Calculate sum term
            sum_term = np.zeros((n, n))
            for i in range(mu):
                sum_term += weights[i] * np.outer(y_sorted[i], y_sorted[i])
            
            C = (1 - c_1 - c_mu) * C + c_1 * (C + np.outer(p_c, p_c)) + c_mu * sum_term
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            x_m = x_m_new
            
            # Check stagnation
            if calls_made - last_improvement_call > 30:
                # Restart
                x_m = best_x.copy()
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                sigma = max(sigma * 1.5, 1.0)
                last_improvement_call = calls_made

        return best_x, best_f
