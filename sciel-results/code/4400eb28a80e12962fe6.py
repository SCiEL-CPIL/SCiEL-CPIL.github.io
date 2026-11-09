"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # 50s safety margin before 330s kill
        
        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * ln(n))
        # For n=5, lambda = 4 + floor(3 * 1.609) = 4 + 4 = 8
        # Let's use a slightly larger population for robustness: 10 or 12
        # Standard recommendation is lambda = 4 + 3*ln(n)
        lam = int(4 + 3 * np.log(self.dim))
        if lam < 4:
            lam = 4
        
        # Initial mean: center of the box, or random within [-4, 4]
        # The problem states minimizer is in [-4, 4]^5. 
        # Starting at 0 is a reasonable heuristic for BBOB functions which are often symmetric or centered.
        x_mean = np.zeros(self.dim)
        
        # Initial sigma: 
        # The box is [-5, 5], so width is 10.
        # A good initial sigma is often (x_max - x_min) / 10 or similar.
        # Let's try 1.0 or 2.0. 
        # BBOB functions usually have a scale. Let's start with 1.0.
        sigma = 1.0
        
        # CMA-ES weights and parameters
        mu = lam // 2
        # Weights: log-linear
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # Initial covariance matrix: Identity
        C = np.eye(self.dim)
        # Initial evolution paths
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Precompute sqrt of C
        D, B = np.linalg.eigh(C)
        D = np.sqrt(D)
        # B is the matrix of eigenvectors
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        calls_made = 1
        
        # Main loop
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = np.zeros((lam, self.dim))
            for i in range(lam):
                # Sample from N(0, I)
                z = np.random.randn(self.dim)
                # Transform to N(0, C)
                # x = x_mean + sigma * B * (D * z)
                ys[i] = x_mean + sigma * (B * (D * z))
            
            # Evaluate population
            fs = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                fs[i] = func(ys[i])
                calls_made += 1
            
            # If we broke out of the evaluation loop due to time, break the main loop
            if time.time() > deadline:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Update mean
            x_new = x_mean + sigma * np.dot(weights, ys[:mu])
            
            # Update evolution path for covariance matrix
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (1/sigma) * (x_new - x_mean)
            # Note: The term (x_new - x_mean) is the step in the transformed space?
            # Actually, the standard update is:
            # p_c = (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) / sigma * (x_new - x_mean)
            # Wait, x_new - x_mean is the step in the original space.
            # The update rule in CMA-ES is:
            # p_c <- (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) * (x_new - x_mean) / sigma
            # This p_c is in the original space.
            
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            
            # Update evolution path for step size
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * h_sigma * B^{-1} D^{-1} (x_new - x_mean) / sigma
            # h_sigma is a hysteresis operator to avoid premature convergence of sigma
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2 gen}) < (1.4 + 2/(dim+1)) * E[N(0, I)]
            
            # Calculate h_sigma
            # E[N(0, I)] = sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim^2))
            # Let's approximate E[N(0, I)] as sqrt(dim)
            E_norm = np.sqrt(self.dim)
            # gen is the current generation count
            # We need to track generations
            # Let's add a generation counter
            
            # For simplicity and robustness in this short budget, we can skip the h_sigma check 
            # or implement it simply.
            # Let's implement it properly.
            # We need a generation counter.
            # Let's assume we are at generation g.
            # The formula for h_sigma depends on the history.
            # A simpler approach: just update p_sigma without h_sigma if it's complex, 
            # but h_sigma is important for performance.
            
            # Let's use a simpler update for p_sigma if we don't track generations easily.
            # Actually, we can just track generations.
            
            # Let's restart the loop structure to include generation count.
            # This code block is getting complex. Let's refine.
            
            # Update p_sigma
            # We need the inverse of the matrix B * diag(D)
            # inv_BD = np.linalg.inv(B * np.diag(D))
            # This is expensive to compute every step.
            # Alternatively, we can update p_sigma in the transformed space.
            
            # Let's stick to the standard implementation which computes the inverse or uses eigen decomposition updates.
            # Given the small dimension (5), computing inverse is cheap.
            
            # However, for this short budget (500 calls), a simplified CMA-ES might be sufficient.
            # Let's try a simpler (1+lambda)-ES with covariance adaptation if full CMA-ES is too complex to write correctly in one go.
            # But CMA-ES is much better.
            
            # Let's write a cleaner CMA-ES implementation.
            pass # Placeholder for logic refinement

# The above thinking process is a bit messy. Let's write a clean, correct CMA-ES implementation.

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280  # 50s safety margin before 330s kill
        
        n = self.dim
        # Population size
        lam = int(4 + 3 * np.log(n))
        if lam < 4:
            lam = 4
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Eigen decomposition
        D, B = np.linalg.eigh(C)
        D = np.sqrt(D)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1
        gen = 0
        
        # Precompute constants
        E_norm = np.sqrt(n) * (1 - 1/(4*n) - 1/(25*n*n))
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
                
            gen += 1
            # Generate population
            ys = np.zeros((lam, n))
            for i in range(lam):
                z = np.random.randn(n)
                ys[i] = x_mean + sigma * (B * (D * z))
            
            # Evaluate
            fs = np.zeros(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                fs[i] = func(ys[i])
                calls_made += 1
            
            if time.time() > deadline:
                break
            
            # Sort
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Update mean
            x_new = x_mean + sigma * np.dot(weights, ys[:mu])
            
            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            
            # Update p_sigma
            # h_sigma check
            # ||p_sigma|| / sqrt(1 - (1 - c_sigma)^(2 * gen)) < (1.4 + 2/(n+1)) * E_norm
            # We need the norm of p_sigma in the transformed space?
            # The update is: p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * h_sigma * B^{-1} D^{-1} (x_new - x_mean) / sigma
            
            # Compute step in transformed space
            step_orig = (x_new - x_mean) / sigma
            # Transform to standard normal space
            # x = B D z => z = D^{-1} B^{-1} x
            # B is orthogonal, so B^{-1} = B.T
            step_trans = np.linalg.solve(B.T, step_orig / D)
            
            # h_sigma
            # We need the norm of p_sigma. p_sigma is in the transformed space?
            # In standard CMA-ES, p_sigma is maintained in the transformed space (standard normal space).
            # Let's assume p_sigma is in the transformed space.
            # Then the update is:
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * h_sigma * step_trans
            
            # Check h_sigma
            # The condition is based on the norm of p_sigma.
            # norm_p_sigma = np.linalg.norm(p_sigma)
            # threshold = (1.4 + 2/(n+1)) * E_norm
            # gen_factor = np.sqrt(1 - (1 - c_sigma)**(2 * gen))
            # h_sigma = 1 if norm_p_sigma / gen_factor < threshold else 0
            
            # However, gen_factor approaches 1.
            # Let's use a simpler heuristic or just set h_sigma = 1 for now to avoid complexity, 
            # or implement it correctly.
            
            # Correct implementation:
            # p_sigma is in the transformed space.
            # norm_p_sigma = np.linalg.norm(p_sigma)
            # threshold = (1.4 + 2/(n+1)) * E_norm
            # # Approximate the denominator as 1 for large gen, or calculate it.
            # # For small gen, it's small, so h_sigma is likely 0? No, if norm is small, h_sigma=1.
            # # Actually, if the path is short, we want to keep sigma.
            # # Let's just set h_sigma = 1.0 for simplicity in this short budget.
            # # Or better, calculate it.
            # gen_factor = np.sqrt(1 - (1 - c_sigma)**(2 * gen))
            # if gen_factor == 0:
            #     h_sigma = 1.0
            # else:
            #     h_sigma = 1.0 if norm_p_sigma / gen_factor < threshold else 0.0
            
            # Let's implement the h_sigma check properly.
            norm_p_sigma = np.linalg.norm(p_sigma)
            threshold = (1.4 + 2/(n+1)) * E_norm
            gen_factor = np.sqrt(1 - (1 - c_sigma)**(2 * gen))
            if gen_factor == 0:
                h_sigma = 1.0
            else:
                h_sigma = 1.0 if norm_p_sigma / gen_factor < threshold else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * h_sigma * step_trans
            
            # Update sigma
            # sigma = sigma * exp(c_sigma/d_sigma * (||p_sigma||/E_norm - 1))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E_norm - 1))
            
            # Update C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_c) c_c (2 - c_c) I) + c_mu sum w_i (y_i y_i^T)
            # y_i are the steps in the original space? No, y_i in the formula are the normalized steps?
            # The formula uses the steps in the original space divided by sigma?
            # Let's check the standard CMA-ES update for C.
            # C <- (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_c) c_c (2 - c_c) I) + c_mu sum_{i=1}^mu w_i (z_i z_i^T)
            # where z_i = (x_i - x_old) / sigma
            # p_c is in the original space.
            # z_i are in the original space, scaled by 1/sigma.
            
            # Calculate h_c
            # h_c = 1 if ||p_c|| / sqrt(1 - (1 - c_c)^(2 * gen)) < (1.4 + 2/(n+1)) * sqrt(c_c (2 - c_c)) * E_norm else 0
            # Note: The threshold for h_c is different from h_sigma.
            # threshold_c = (1.4 + 2/(n+1)) * np.sqrt(c_c * (2 - c_c)) * E_norm
            # gen_factor_c = np.sqrt(1 - (1 - c_c)**(2 * gen))
            # norm_p_c = np.linalg.norm(p_c)
            # h_c = 1.0 if norm_p_c / gen_factor_c < threshold_c else 0.0
            
            # Let's calculate h_c
            norm_p_c = np.linalg.norm(p_c)
            threshold_c = (1.4 + 2/(n+1)) * np.sqrt(c_c * (2 - c_c)) * E_norm
            gen_factor_c = np.sqrt(1 - (1 - c_c)**(2 * gen))
            if gen_factor_c == 0:
                h_c = 1.0
            else:
                h_c = 1.0 if norm_p_c / gen_factor_c < threshold_c else 0.0
            
            # Update C
            # The term c_1 (p_c p_c^T + (1 - h_c) c_c (2 - c_c) I)
            # The term c_mu sum w_i z_i z_i^T
            # z_i = (x_i - x_mean_old) / sigma
            
            z_matrix = (ys[:mu] - x_mean) / sigma
            # Sum of weighted outer products
            # C_update = c_mu * sum(w_i * np.outer(z_i, z_i))
            C_update = np.zeros((n, n))
            for i in range(mu):
                C_update += weights[i] * np.outer(z_matrix[i], z_matrix[i])
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * np.eye(n)) + c_mu * C_update
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update eigen decomposition
            # We can do this every step for small n
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 0)) # Ensure non-negative
            
            # Update mean
            x_mean = x_new
            
        return best_x, best_f

# EVOLVE-BLOCK-END
