"""Strategy: CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 320.0

        # CMA-ES parameters
        # lambda (population size)
        lambda_pop = 4 + int(3 * np.log(dim))
        
        # Initial mean
        # Start near center, slightly randomized to avoid symmetry breaking issues if any
        rng = np.random.default_rng(42)
        m = np.zeros(dim)
        
        # Initial step size
        sigma = 2.0
        
        # Initial covariance
        C = np.eye(dim)
        
        # Pre-compute constants
        mu = lambda_pop // 2  # number of selected parents
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Chi-squared correction
        chi_lambda = np.sqrt(lambda_pop) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # State
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Evaluation history
        best_x = m.copy()
        best_fx = float('inf')
        
        # Initial population evaluation
        # Generate first population
        # We need to ensure we don't exceed budget
        # Total budget is 300. lambda_pop is roughly 13 for dim=5.
        # 300 / 13 is about 23 generations.
        
        calls_made = 0
        
        for gen in range(budget):
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            # B = C^{1/2}
            # Use eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive
            eigvals = np.maximum(eigvals, 1e-10)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            Y = rng.standard_normal((lambda_pop, dim))
            X = m + sigma * (Y @ C_sqrt.T)
            
            # Evaluate
            F = []
            for x in X:
                if time.time() > deadline:
                    break
                fx = func(x)
                F.append(fx)
                calls_made += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            
            if len(F) < lambda_pop:
                break
                
            F = np.array(F)
            X = np.array(X[:len(F)])
            
            # Sort by fitness
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update mean
            m_new = np.sum(w[:, np.newaxis] * X[:mu], axis=0)
            
            # Update covariance
            # y = (m_new - m) / sigma
            y = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            # Hessian of log det C
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) - 1 + 1e-10) / (1 + c_sigma)
            h_sigma = min(1.0, h_sigma)
            
            # Delta term for covariance update
            delta_term = c_1 * (np.outer(p_c, p_c) - C) + c_mu * np.sum(w[:, np.newaxis] * np.outer(y, y), axis=0)
            
            # Correction term
            correction = c_1 * (1 - c_1) * (np.outer(p_c, p_c) - C) + c_mu * (1 - c_mu) * np.sum(w[:, np.newaxis] * np.outer(y, y), axis=0)
            
            # Actually, standard CMA update is:
            # C = (1 - c_1 - c_mu + c_1 * (1 - h_sigma) * (1 - 2 * c_sigma) + c_mu * (1 - h_sigma) * (1 - 2 * c_sigma)) * C 
            #     + c_1 * (1 + (2/h_sigma - 1) * h_sigma) * p_c p_c^T 
            #     + c_mu * sum w_i (y_i - h_sigma * y_bar)(y_i - h_sigma * y_bar)^T
            # Let's use the standard formulation from Hansen's paper
            # C = (1 - c1 - cmu) * C + c1 * (1 + (2/hsig - 1)*hsig) * p_c p_c^T + cmu * sum wi (yi - hsig*ybar)(yi - hsig*ybar)^T
            # Wait, the h_sigma term is usually:
            # C = (1 - c1 - cmu + c1*(1-hs)*(1-2cs) + cmu*(1-hs)*(1-2cs)) * C + ...
            # Let's stick to the simpler, robust version often used in implementations:
            
            # Recalculate C update carefully
            # y_bar is the weighted mean of y, which is m_new - m / sigma ? No, y is defined as (m_new - m)/sigma.
            # The weighted mean of the selected y's is exactly y.
            # So the term is sum wi * (yi - y) (yi - y)^T
            
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (1 + (2.0 / h_sigma - 1.0) * h_sigma) * np.outer(p_c, p_c)
            C_new += c_mu * np.sum(w[:, np.newaxis] * np.outer(X[:mu] - m, X[:mu] - m), axis=0) / (sigma * sigma)
            
            # Symmetrize
            C_new = (C_new + C_new.T) / 2.0
            
            # Ensure positive definite
            # Add small regularization if needed
            C = C_new
            
            m = m_new
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_lambda * (np.linalg.norm(p_sigma) / chi_lambda - 1))
            
            # Optional: Rescale if sigma gets too big or small
            if sigma < 1e-6:
                break
            if sigma > 10.0:
                sigma = 10.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
