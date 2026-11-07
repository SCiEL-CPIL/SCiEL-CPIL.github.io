"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = max(4 + int(3 * np.log(n)), 10) # At least 10 for n=5
        # Learning rates
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size adaptation
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initial mean and step size
        # Uniform initialization is safe, but CMA-ES works best with a reasonable initial sigma
        # Box is [-5, 5], width 10.
        # Start at origin? The problem says minimizer is in [-4,4]. Origin is a good guess.
        # However, to be robust against shifts, a uniform start is safer for the first few steps, 
        # but CMA-ES needs a center. Let's start at 0 with a large sigma.
        m = np.zeros(n)
        sigma = 3.0 # Covers most of the box
        
        # State variables
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        hs = np.sqrt((1 + c_c) / (1 - c_c)) + 2
        
        # Budget tracking
        calls = 0
        best_x = None
        best_f = np.inf
        
        # Initial population evaluation to set a baseline
        # We will generate the first population now.
        
        while calls + lam <= budget:
            # Sample population
            B = np.linalg.cholesky(C)
            # Generate normal samples
            Z = np.random.randn(lam, n)
            # Transform to CMA-ES samples
            # x = m + sigma * B @ z
            pop = m + sigma * (Z @ B.T)
            
            # Evaluate
            fvals = np.array([func(x) for x in pop])
            calls += lam
            
            # Update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = pop[idx_min].copy()
                
            # Sort by fitness
            order = np.argsort(fvals)
            x_sorted = pop[order]
            f_sorted = fvals[order]
            
            # Weighted recombination
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * x_sorted[i]
            m_new = m + sigma * y_w
            
            # Check for convergence (optional, but good for stopping early if stuck)
            # If sigma is very small, break
            if sigma < 1e-8:
                break
                
            # Path updates
            y_sigma = np.linalg.inv(B) @ y_w
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (2 * n**2) + 1 / (8 * n**4))
            
            if np.linalg.norm(p_sigma) / hs < (1.4 + 2 / (n + 1)) * chi_n:
                c_sigma = c_sigma # No change in effective c_sigma usually, but logic for sigma update
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Rank-one and rank-mu updates
            delta_hs = (1 - c_c) ** 2 - (1 - c_c) ** 2 # This part of standard CMA-ES is complex, 
            # let's use the standard update formula:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - (1-c_c)^2) * C) + c_mu * sum w_i y_i y_i^T
            # Note: The term (1 - (1-c_c)^2) * C is often written as (1 - (1-c_c)^2) * C
            # Actually, the standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - (1-c_c)^2) * C) + c_mu * sum w_i y_i y_i^T
            # Wait, the term is usually:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - (1-c_c)^2) * C) + ...
            # Let's use a simplified but robust version or standard library if available? 
            # No, must implement.
            
            # Correct standard update:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c.T + (1 - (1 - c_c)**2) * C) + c_mu * sum(weights[i] * y_i @ y_i.T)
            # Note: y_i here are the transformed vectors (B^-1 (x_i - m_old))
            
            y_list = []
            for i in range(mu):
                # y_i = B^-1 (x_sorted[i] - m)
                # We have Z_sorted = Z[order[:mu]]
                # x_sorted[i] = m + sigma * B @ Z_sorted[i]
                # So (x_sorted[i] - m) / sigma = B @ Z_sorted[i]
                # y_i = B^-1 @ (B @ Z_sorted[i]) = Z_sorted[i]
                # This simplifies things! The y vectors for the update are just the sorted Z vectors.
                y_list.append(Z[order[i]])
                
            C_update = np.zeros((n, n))
            for i in range(mu):
                C_update += weights[i] * np.outer(y_list[i], y_list[i])
                
            # Calculate the delta_hs term properly
            # The term (1 - (1-c_c)^2) * C is equivalent to (2*c_c - c_c^2) * C
            # But often it's just kept as is.
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * C_update
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Check eigenvalues, if any are negative, add identity
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += (1e-12 - np.min(eigvals)) * np.eye(n)
                
            m = m_new
            
            # Clip mean to bounds to keep search in the box
            # If mean goes out, reflect or clip? Clipping is safer for CMA-ES stability
            m = np.clip(m, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
