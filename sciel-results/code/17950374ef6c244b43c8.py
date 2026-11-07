"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Time limit guard: evaluator kills at 330s, we aim for 280s safety
        start_time = time.time()
        time_limit = 280.0

        # CMA-ES Parameters
        # Population size lambda
        # Standard formula: 4 + floor(3 * ln(n))
        # For n=5, lambda = 4 + 4 = 8. Let's use 9 for slightly better exploration.
        lam = 9
        
        # Initial mean: Center of the box
        m = np.zeros(n)
        
        # Initial step size
        # The box is [-5, 5], so width is 10.
        # A common heuristic is 0.3 * range, but for CMA-ES, 
        # starting with a step size that covers the box is good for exploration.
        # However, CMA-ES adapts quickly. 
        # Let's start with sigma = 2.0 (covers ~95% of a dimension of width 10? No, 2*2=4 range).
        # If we want to explore the whole box, sigma should be larger.
        # But the optimum is in [-4, 4]^5. 
        # Let's try sigma = 1.5.
        sigma = 1.5
        
        # Covariance matrix
        C = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Weights
        # Standard CMA-ES weights
        mu = lam // 2
        weights = np.log(lam / 2 + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1) - 1)) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Best solution tracking
        best_x = m.copy()
        best_f = float('inf')
        
        # Evaluate initial mean
        f_m = func(m)
        if f_m < best_f:
            best_f = f_m
            best_x = m.copy()
        
        calls_used = 1
        
        # Main loop
        while calls_used < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
                
            # Generate population
            B = np.linalg.cholesky(C)
            
            # Sample from standard normal
            Z = np.random.randn(lam, n)
            
            # Transform to search space
            Y = np.array([m + sigma * (Z[i] @ B) for i in range(lam)])
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls_used += lam
            
            # Update best
            for i in range(lam):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = Y[i].copy()
            
            # Check time again after evaluation batch
            if time.time() - start_time > time_limit:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Weighted recombination
            m_new = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (weights @ (B.T @ (Y[:mu] - m).T).T).T) / sigma
            # Note: The standard formula is:
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B.T @ (weights @ (B.T @ (Y - m).T).T).T)
            # Let's simplify the calculation:
            # D = B.T @ (weights @ (B.T @ (Y[:mu] - m).T).T).T
            # Actually, let's use the standard vectorized form:
            # y_w = weights @ (Y[:mu] - m)  # This is not quite right, Y is (mu, n)
            
            # Correct update for p_sigma:
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (np.dot(weights, (Y[:mu] - m) @ B.T))) / sigma
            # Let's be careful with dimensions.
            # Y[:mu] is (mu, n). m is (n,).
            # Y[:mu] - m is (mu, n).
            # (Y[:mu] - m) @ B.T is (mu, n).
            # weights is (mu,).
            # weights @ (Y[:mu] - m) @ B.T is (n,).
            # B.T @ ( ... ) is (n,).
            
            diff = Y[:mu] - m
            weighted_diff = np.dot(weights, diff) # (n,)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ weighted_diff) / sigma
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ (np.dot(weights, diff))) / sigma
            # Wait, p_c uses the untransformed difference? No, p_c is in the transformed space?
            # Standard CMA-ES:
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ (weights @ (B.T @ (Y[:mu] - m).T).T).T) / sigma
            # This is the same term as in p_sigma but with different weights?
            # No, the term is:
            # y_c = (1/sigma) * B.T @ (weights @ (B[:mu] @ B.T))? No.
            # Let's stick to the standard implementation from Hansen.
            
            # Recalculate p_sigma and p_c correctly
            # y_w = weights @ (Y[:mu] - m)  -> (n,)
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ y_w) / sigma
            
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ y_w) / sigma
            # Wait, is the term for p_c the same?
            # Yes, in the standard CMA-ES, the recombination vector is the same, but the paths are updated with different rates.
            # However, usually p_c is updated with the *mean shift* in the transformed space?
            # Let's check the reference.
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) / sigma * B^T (w^T (Y - m) B^T)
            # p_c = (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) / sigma * B^T (w^T (Y - m) B^T)
            # The term is indeed the same vector. Let's call it `y_trans`.
            
            y_trans = (B.T @ weighted_diff) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_trans
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_trans
            
            # Update C
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used + 1)))) / chi_n < (1 + 2 / (n + 1)) * (1 - c_sigma)
            
            h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (calls_used + 1)))) / chi_n < (1 + 2 / (n + 1)) * (1 - c_c)
            
            if h_sigma:
                C = (1 - c_1 - c_mu + (1 - c_1) * (1 - h_sigma) * d_sigma) * C + c_1 * np.outer(p_sigma, p_sigma) + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(Y[:mu] - m, Y[:mu] - m) / sigma**2, axis=0)
            else:
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(Y[:mu] - m, Y[:mu] - m) / sigma**2, axis=0)
                
            # Ensure symmetry and positive semi-definiteness
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            m = m_new
            
            # Evaluate new mean? No, CMA-ES doesn't usually evaluate the mean explicitly unless it's part of the population.
            # But for tracking best, we already did.
            # However, evaluating the new mean can be useful if it's better.
            # But we have a budget constraint. The population evaluation is the main cost.
            # We can skip evaluating m_new explicitly to save budget, as it is a weighted average of evaluated points.
            # Actually, the new mean is NOT necessarily one of the evaluated points.
            # But its value is unknown.
            # In standard CMA-ES, we don't evaluate the new mean separately.
            # We just use it for the next generation.
            
            # Check if we should restart?
            # If sigma is too small or too large, we might restart.
            # But for 500 budget, a single run is likely sufficient.
            
            # Safety: if sigma becomes extremely small, break
            if sigma < 1e-10:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
