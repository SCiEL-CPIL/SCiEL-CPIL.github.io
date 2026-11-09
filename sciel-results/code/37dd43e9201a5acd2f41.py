"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Initial mean: center of the box
        mean = np.zeros(dim)
        
        # Initial step size: cover a significant portion of the box
        # Box is [-5, 5], so width is 10. 
        # A good initial sigma is roughly box_size / 4 or so.
        sigma = 2.0
        
        # Population size
        # Standard is 4 + floor(3 * log(N))
        # For N=5, 4 + 3*1.6 = 8.8 -> 8 or 9. Let's use 10 for better exploration.
        pop_size = 10
        lambda_ = pop_size
        mu = lambda_ // 2  # Weights for top half
        
        # Weights
        weights = np.ones(mu)
        weights **= 1.0  # Linear weights (or could be log)
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        # Step size control
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Covariance matrix adaptation
        c_c = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        
        # Initial covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        D = 1.0
        B = np.eye(dim)
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        calls_made = 0
        best_x = mean.copy()
        best_f = np.inf
        
        # Helper to decompose C
        def update_C_inv(C):
            nonlocal C_inv, D, B
            # Eigen decomposition
            D_eig, B_eig = np.linalg.eigh(C)
            # Ensure positive definite
            D_eig = np.maximum(D_eig, 1e-10)
            D = np.sqrt(D_eig)
            B = B_eig
            C_inv = (B / D) @ (B.T / D)
            
        update_C_inv(C)
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = np.zeros((lambda_, dim))
            xs = np.zeros((lambda_, dim))
            fs = np.zeros(lambda_)
            
            for i in range(lambda_):
                if calls_made >= budget or time.time() > deadline:
                    break
                z = np.random.standard_normal(dim)
                # y = B @ (D * z)
                y = (B * D) @ z
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                
                fx = func(x)
                calls_made += 1
                ys[i] = y
                xs[i] = x
                fs[i] = fx
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            xs = xs[order]
            fs = fs[order]
            
            # Update mean
            mean_new = sum(weights[i] * xs[i] for i in range(mu))
            
            # Check for stagnation/convergence
            if np.linalg.norm(mean_new - mean) < 1e-12:
                # If mean doesn't move, we might be stuck.
                # In CMA-ES, if sigma is very small, we stop.
                if sigma < 1e-12:
                    break
            else:
                mean = mean_new
            
            # Update p_sigma
            # p_sigma = (1-c_sigma)*p_sigma + c_sigma/sqrt(c_sigma*(2-c_sigma)) * (B^T D^-1 / sqrt(mu_eff)) (x_mean - x_mean_old)
            # We need the old mean for this, so we should have stored it.
            # Let's refactor to store old_mean before updating.
            
            # Actually, let's just implement the standard update carefully.
            # The standard update uses the difference between the new mean and the old mean.
            # Since we updated 'mean' in place above, we lost the old mean.
            # Let's restart the loop logic with proper state tracking.
            
            # To keep it simple and robust within the time limit, let's use a simplified CMA-ES 
            # or just rely on the fact that even a basic version is better than NM.
            # However, the "Different Powers" function is not rotated, so CMA-ES will adapt quickly.
            
            # Let's do the updates properly.
            # We need old_mean.
            # I will re-implement the loop body to be cleaner.
            
            # For this specific implementation, I will simplify the CMA-ES updates to avoid 
            # complex matrix algebra errors in a short script, but keep the core adaptation.
            
            # Re-calculate p_sigma update using the difference
            # Note: This is a slightly simplified version for robustness in the short code block.
            # Full CMA-ES is preferred, but let's ensure it runs.
            
            # Update p_c
            # p_c = (1-c_c)*p_c + sqrt(c_c*(2-c_c)*mueff) * (B^T D^-1 / sigma) (x_mean - x_mean_old)
            
            # Since we don't have old_mean, let's just use the change in mean.
            # Actually, let's just use a simpler adaptive strategy if CMA-ES becomes too complex to verify quickly.
            # But CMA-ES is the strategy. Let's stick to it but be careful.
            
            # Let's assume the previous 'mean' was the old mean.
            # I will rewrite the loop to be safer.
            
            # Break out and use a simpler, robust CMA-ES implementation below.
            break 

        # The above loop had a state issue. Let's provide a clean, complete CMA-ES implementation.
        # I will restart the logic in a cleaner way.
        
        # --- Clean CMA-ES Implementation ---
        mean = np.zeros(dim)
        sigma = 2.0
        pop_size = 10
        mu = pop_size // 2
        weights = np.ones(mu)
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        c_c = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute B and D
        B = np.eye(dim)
        D = np.ones(dim)
        
        calls_made = 0
        best_x = mean.copy()
        best_f = np.inf
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate samples
            ys = []
            xs = []
            fs = []
            
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break
                z = np.random.standard_normal(dim)
                y = (B * D) @ z
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                ys.append(y)
                xs.append(x)
                fs.append(fx)
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if len(xs) < 2:
                break
                
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # Sort
            order = np.argsort(fs)
            ys = ys[order]
            xs = xs[order]
            fs = fs[order]
            
            # Update mean
            mean_old = mean.copy()
            mean = sum(weights[i] * xs[i] for i in range(mu))
            
            # Update p_sigma
            if np.linalg.norm(mean - mean_old) > 1e-14:
                p_sigma = (1 - c_sigma) * p_sigma + \
                          np.sqrt(c_sigma * (2 - c_sigma) * mueff) / (sigma * np.linalg.norm(mean - mean_old)) * \
                          (B.T @ (D * (B.T @ (mean - mean_old)))) # This is complex, let's simplify
                # Simplified p_sigma update:
                # y_mean = (mean - mean_old) / sigma
                # p_sigma += c_sigma * (B^T D^-1 y_mean) / ||B^T D^-1 y_mean|| * h_sigma
                # For simplicity and robustness in this short script, we can skip p_sigma adaptation 
                # and just use a fixed c_sigma or a simpler rule, as the covariance update is the main driver.
                # However, sigma adaptation is important.
                # Let's use the standard update but computed carefully.
                y_mean = (mean - mean_old) / sigma
                # Project to B^T D^-1 space
                # p_sigma update vector: v = B^T (D^-1 * (B^T y_mean)) ? No.
                # The update is: p_sigma <- (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mueff) * B^T D^-1 / sigma * (x_mean - x_mean_old)
                # Let's compute B^T D^-1 (mean - mean_old) / sigma
                # D is diagonal. D^-1 is 1/D.
                # B^T @ ((mean - mean_old) / (sigma * D)) ? No, (B * D) @ z = y. So z = D^-1 B^T y.
                # The direction in z-space is z_mean = D^-1 B^T (mean - mean_old) / sigma.
                # p_sigma is in z-space? No, p_sigma is in the rotated space.
                # Let's stick to the standard algorithm definition.
                
                # Actually, to ensure correctness and avoid subtle bugs in a 300s run,
                # I will use a simplified CMA-ES that only adapts Sigma and the Covariance Matrix C,
                # skipping the path length adaptation for sigma (using a fixed c_sigma effect or simpler rule).
                # Or better, just use the standard update formulas correctly.
                
                # Let's use the formulas from Hansen's paper.
                # p_sigma <- (1-c_sigma)*p_sigma + c_sigma/sqrt(c_sigma*(2-c_sigma)) * (B^T D^-1 / sigma) * (x_mean - x_mean_old)
                # Note: B^T D^-1 is the inverse of the transformation from z to y.
                # y = B D z. So z = D^-1 B^T y.
                # So the vector in z-space is D^-1 B^T (mean - mean_old) / sigma.
                
                z_mean = (B.T * (1.0/D)) @ ((mean - mean_old) / sigma)
                
                # h_sigma: Heaviside function
                # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2gen}) < (1.4 + 2/(dim+1)) * chi_n
                # We don't track gen, so we approximate.
                # For now, let's just update p_sigma and use it to adapt sigma.
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_mean
                
                # Adapt sigma
                # sigma <- sigma * exp(c_sigma/d_sigma * (||p_sigma||/chi_n - 1))
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                # Clamp sigma
                sigma = np.clip(sigma, 1e-12, 10.0)
            
            # Update p_c
            # p_c <- (1-c_c)*p_c + sqrt(c_c(2-c_c)mueff) * h_sigma * B^T D^-1 / sigma * (x_mean - x_mean_old)
            # h_sigma is typically 1 if the step is accepted.
            h_sigma = 1.0 # Simplified
            z_mean = (B.T * (1.0/D)) @ ((mean - mean_old) / sigma)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * h_sigma * z_mean
            
            # Update C
            # C <- (1 - c_1 - c_mu + mu_delta)*C + c_1 * (p_c p_c^T + (1-h_sigma)c_c(2-c_c)I) + c_mu * sum(wi yi yi^T)
            # mu_delta = c_mu^2 / (c_1 + c_mu)
            mu_delta = c_mu**2 / (c_1 + c_mu)
            
            # Compute the rank-1 and rank-mu updates
            # y_i are the selected steps (in y-space, i.e., B D z)
            # We need the y_i for the top mu individuals.
            # We have ys (sorted).
            y_selected = ys[:mu]
            
            # Rank-mu term: sum wi * yi * yi^T
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y_selected[i], y_selected[i])
                
            # Rank-1 term
            rank_1 = np.outer(p_c, p_c)
            
            # Update C
            C = (1 - c_1 - c_mu + mu_delta) * C + c_1 * (rank_1 + (1 - h_sigma) * c_c * (2 - c_c) * np.eye(dim)) + c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Check eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                # Add small regularization
                C += 1e-10 * np.eye(dim)
                
            # Update B and D
            D_eig, B_eig = np.linalg.eigh(C)
            D_eig = np.maximum(D_eig, 1e-10)
            D = np.sqrt(D_eig)
            B = B_eig
            
            # Convergence check
            if sigma < 1e-12:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
