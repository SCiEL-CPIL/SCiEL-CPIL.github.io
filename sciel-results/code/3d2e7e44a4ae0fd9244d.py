"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Population size: 4 + floor(3 * ln(n))
        lam = 4 + int(3 * np.log(dim))
        if lam < 2:
            lam = 2
        
        # Initial sigma
        # Box is [-5, 5], center 0. Minimizer in [-4, 4].
        # Start with a sigma that covers a good portion of the box.
        sigma = 2.0 
        
        # Initial mean: center of the box
        mean = np.zeros(dim)
        
        # Covariance matrix: identity
        C = np.eye(dim)
        
        # Inverse square root of C for sampling
        C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C))
        
        # Strategy parameters
        mu = lam // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Step size control
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        d_sigma = c_sigma * (1 + 0.5 * min(1, c_sigma - 1) * (mu_eff / dim + 1))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = mean.copy()
        best_f = func(best_x)
        
        # Budget tracking
        evals_used = 1
        max_evals = budget
        
        # Main loop
        while evals_used < max_evals:
            # Generate population
            # Ensure we don't exceed budget in this generation
            # If remaining budget < lam, we can't do a full generation, 
            # but CMA-ES usually works in full generations. 
            # Given 500 budget and lam=10, we get 49 full generations + 9 extra evals?
            # It's better to just run until we hit the budget.
            
            if max_evals - evals_used < lam:
                break
                
            # Sample candidates
            candidates = []
            f_values = []
            
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = C_sqrt_inv @ z
                x_new = mean + sigma * y
                # Clip to box
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = func(x_new)
                evals_used += 1
                
                candidates.append(x_new)
                f_values.append(f_new)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            
            # Sort candidates by fitness
            idx = np.argsort(f_values)
            x_sorted = np.array(candidates)[idx]
            f_sorted = np.array(f_values)[idx]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update step size control
            y_mean = (mean - old_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            
            # Hessian update for covariance
            # We need the inverse square root of C to transform y_mean for p_c
            # Actually, standard CMA-ES uses:
            # p_c = (1-c_c)*p_c + h_sigma * sqrt(c_c*(2-c_c)*mu_eff) * C_sqrt_inv @ y_mean
            # But C_sqrt_inv changes. 
            # Simplified: We update C directly.
            
            # Check for sigma adaptation
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (max_evals - evals_used))) < (1.4 + 2 / (dim + 1)) * chi_n:
                h_sigma = 1
            else:
                h_sigma = 0
                
            sigma = sigma * np.exp(d_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance
            # Delta h term
            delta_h = (1 - h_sigma) * c_c * (2 - c_c) * np.linalg.norm(p_c) ** 2 / dim
            # Wait, standard formula for p_c update involves C_sqrt_inv.
            # Let's stick to a simpler, robust implementation often used in practice:
            # We update C using the rank-1 and rank-mu updates.
            
            # To avoid complexity of maintaining C_sqrt_inv correctly with changing C,
            # we can recompute it or use a simpler update.
            # However, for dim=5, recomputing Cholesky is cheap.
            
            # Let's do the standard updates properly.
            # We need C_sqrt_inv for the p_c update and rank-mu update.
            # Since C changes, C_sqrt_inv changes.
            
            # 1. Update p_c
            # y_mean is in the "transformed" space? No, y_mean is (new_mean - old_mean)/sigma.
            # The update for p_c is:
            # p_c = (1-c_c)*p_c + h_sigma * sqrt(c_c*(2-c_c)*mu_eff) * C_sqrt_inv @ y_mean
            # Note: C_sqrt_inv is the inverse square root of the OLD C.
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt_inv @ y_mean)
            
            # 2. Update C
            # Rank-1 update
            # C = (1 + c_1 * delta_h + c_mu - c_1) * C + c_1 * (p_c @ p_c + (1-h_sigma)*c_c*(2-c_c)*C) 
            # Wait, the formula is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1-h_sigma)*c_c*(2-c_c)*C) + c_mu * sum(w_i * (y_i @ y_i))
            # Where y_i = (x_i - old_mean) / sigma
            
            # Let's compute y_vectors for the selected parents
            y_vectors = (x_sorted[:mu] - old_mean) / sigma
            
            # Rank-mu term
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu_term += weights[i] * np.outer(y_vectors[i], y_vectors[i])
                
            # Rank-1 term
            rank_1_term = np.outer(p_c, p_c)
            if h_sigma == 0:
                rank_1_term += c_c * (2 - c_c) * C
                
            # Combine
            C = (1 - c_1 - c_mu) * C + c_1 * rank_1_term + c_mu * rank_mu_term
            
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            
            # Update C_sqrt_inv
            try:
                L = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add jitter
                C += 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)
                C_sqrt_inv = np.linalg.inv(L)
                
        return best_x, best_f
# EVOLVE-BLOCK-END
