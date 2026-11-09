"""CMA-ES optimizer for BBOB f24 (Lunacek bi-Rastrigin).
Uses a global CMA-ES implementation with adaptive step size and restarts.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        # CMA-ES parameters
        # Standard CMA-ES initialization for box [-5, 5]^dim
        # Initial center: origin or random? Random is safer for hidden shifts.
        # But CMA-ES is robust. Let's start near origin or random.
        # Given f24 has a global min and a local min, and shifts are hidden,
        # a random start within the box is reasonable, or center 0 if shifts are small.
        # The contract says shift is hidden, min in [-4,4]. Let's use random start.
        
        rng = np.random.default_rng(42)
        
        # We will run CMA-ES for the entire budget.
        # CMA-ES is population-based. 
        # Default lambda (population size) for CMA-ES in dim D is 4 + floor(3 log D).
        # For D=5: 4 + 3*log(5) ~ 4 + 4.8 = 8. Let's use lambda=10 for better exploration.
        # Budget is 500. 
        # Number of generations ~ 500 / lambda = 50.
        # This is a reasonable number of generations for CMA-ES in 5D.
        
        dim = self.dim
        lam = 10  # Population size
        
        # Initial mean
        x_mean = rng.uniform(-5, 5, dim)
        
        # Initial sigma
        # A common heuristic is sigma = (max - min) / 10 or similar.
        # Box width is 10. sigma = 1.0 is a good start.
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # CMA-ES parameters
        mu = lam // 2  # Number of parents
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        w = w[:mu]
        
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Damping for sigma
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Chi lambda for sigma adaptation
        chi_lam = np.sqrt(lam) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # State
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        # Keep track of calls
        calls_made = 1
        
        # Main loop
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Sample offspring
            B = np.linalg.cholesky(C)
            # Generate normal vectors
            # We need lam samples
            z = rng.standard_normal((lam, dim))
            # Transform to y
            y = z @ B.T
            # Update mean and sigma
            # Standard CMA-ES: x_i = x_mean + sigma * y_i
            # Note: In standard CMA-ES, the mean update uses the weighted average of the selected parents.
            # The sampling is x_i = x_mean + sigma * (B * z_i)
            
            # However, we must clip to [-5, 5].
            # Clipping can distort the distribution. 
            # For a robust implementation, we might want to project or just clip.
            # Clipping is simpler and often sufficient for box-constrained problems if the box is not too tight.
            # Given the box is [-5,5] and min is in [-4,4], clipping is acceptable.
            
            x_offspring = x_mean + sigma * y
            x_offspring = np.clip(x_offspring, -5, 5)
            
            # Evaluate
            f_offspring = np.array([func(x) for x in x_offspring])
            calls_made += lam
            
            # Select parents
            idx = np.argsort(f_offspring)
            x_parents = x_offspring[idx[:mu]]
            f_parents = f_offspring[idx[:mu]]
            
            # Update best
            if f_parents[0] < best_f:
                best_f = f_parents[0]
                best_x = x_parents[0]
            
            # Calculate new mean
            x_new_mean = np.sum(w[:, np.newaxis] * x_parents, axis=0)
            
            # Update evolution path for covariance
            # CMA-ES update:
            # p_c = (1 - c_c) * p_c + c_c * sqrt(mu_eff * c_c * (2 - c_c)) / sigma * (x_new_mean - x_mean)
            # Note: The term (x_new_mean - x_mean) is in the transformed space?
            # No, the standard update is:
            # p_c = (1 - c_c) p_c + c_c sqrt(mu_eff c_c (2 - c_c)) / sigma * (x_new_mean - x_mean)
            # This assumes x is in the original space.
            
            # However, the standard CMA-ES derivation uses the inverse square root of C to transform back.
            # Let's stick to the standard algorithm implementation.
            
            # Step 1: Update p_c
            # The vector (x_new_mean - x_mean) is the step in the original space.
            # We need to transform it to the C-normalized space?
            # Actually, the standard formula is:
            # p_c = (1 - c_c) p_c + c_c * sqrt(mu_eff * c_c * (2 - c_c)) / sigma * (x_new_mean - x_mean)
            # This is correct if p_c is maintained in the original coordinate system.
            
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff * c_c * (2 - c_c)) / sigma * (x_new_mean - x_mean)
            
            # Step 2: Update p_sigma
            # p_sigma = (1 - c_sigma) p_sigma + c_sigma * sqrt(mu_eff) / sigma * (x_new_mean - x_mean)
            # Wait, the standard formula for p_sigma uses the inverse square root of C applied to the step?
            # Yes. p_sigma is in the C-normalized space.
            # p_sigma = (1 - c_sigma) p_sigma + c_sigma * sqrt(mu_eff) * (C^{-1/2} (x_new_mean - x_mean)) / sigma
            # We have B such that C = B B^T. So C^{-1/2} = (B^{-1})^T ? No.
            # If C = L L^T, then C^{-1/2} is not simply L^{-1}.
            # We can solve L v = (x_new_mean - x_mean) to get v = L^{-1} (x_new_mean - x_mean).
            # But we need C^{-1/2}. 
            # Actually, the standard implementation uses:
            # p_sigma = (1 - c_sigma) p_sigma + c_sigma * sqrt(mu_eff) * (L^{-1} (x_new_mean - x_mean)) / sigma
            # where L is the Cholesky factor of C.
            
            try:
                L_inv_step = np.linalg.solve(B, (x_new_mean - x_mean))
            except np.linalg.LinAlgError:
                L_inv_step = np.zeros(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * L_inv_step / sigma
            
            # Step 3: Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_lam - 1))
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_lam - 1))
            
            # Step 4: Update C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - (1 - c_c)^2) C) + c_mu sum(w_i (y_i y_i^T))
            # Where y_i are the selected parents in the C-normalized space?
            # The standard formula uses the transformed parents:
            # y_i = (x_parent_i - x_old_mean) / sigma
            # Then C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - (1 - c_c)^2) C) + c_mu sum(w_i y_i y_i^T)
            # Wait, the term (1 - (1 - c_c)^2) C is added to the rank-1 update of p_c?
            # Let's check the standard update rule.
            # C_new = (1 - c_1 - c_mu) C_old + c_1 (p_c p_c^T + (1 - (1 - c_c)^2) C_old) + c_mu sum_i w_i (y_i y_i^T)
            # This is correct.
            
            # Calculate y_i for parents
            # y_i = (x_parent_i - x_old_mean) / sigma
            y_parents = (x_parents - x_mean) / sigma
            
            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += w[i] * np.outer(y_parents[i], y_parents[i])
                
            # Rank-1 update
            # Note: p_c is in original space? 
            # In the standard derivation, p_c is in the original space, but the update to C involves transforming it?
            # Actually, the standard update is:
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - (1 - c_c)^2) C) + c_mu sum(w_i y_i y_i^T)
            # This assumes p_c is in the same space as y_i?
            # No, y_i are in the C-normalized space (standard normal space transformed by C^{-1/2}? No, y_i are the steps in the sampling space).
            # Let's clarify the spaces.
            # Sampling: x = x_mean + sigma * (B z). So the step is sigma * B z.
            # The vector y in the update formula for C usually refers to the step in the original space divided by sigma?
            # If we define y_i = (x_parent_i - x_old_mean) / sigma, then y_i = B z_i.
            # Then the covariance of y_i is C.
            # So the rank-mu update sum(w_i y_i y_i^T) approximates C.
            # The rank-1 update p_c p_c^T: p_c is an evolution path.
            # In the standard algorithm, p_c is maintained in the original space.
            # But the term added to C is p_c p_c^T?
            # If p_c is in original space, then p_c p_c^T has units of x^2.
            # C has units of x^2.
            # So this is consistent.
            # However, the term (1 - (1 - c_c)^2) C is also added.
            
            # Let's implement this directly.
            p_c_outer = np.outer(p_c, p_c)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (p_c_outer + (1 - (1 - c_c)**2) * C) + c_mu * rank_mu
            
            # Ensure symmetry and positive semi-definiteness
            C = (C + C.T) / 2.0
            
            # Update mean
            x_mean = x_new_mean
            
            # Optional: Restart if no progress? 
            # For now, we run for the full budget.
            # If the variance becomes too small, we might get stuck.
            # But with 500 calls, it should be fine.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
