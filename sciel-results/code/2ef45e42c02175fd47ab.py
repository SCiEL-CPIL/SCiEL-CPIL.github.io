import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin against 330s limit
        
        def time_check():
            return (time.time() - start_time) > timeout

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        # CMA-ES implementation
        # We will use a standard CMA-ES loop.
        # Key parameters for dim=5, budget=500:
        # Lambda (population size) is typically 4 + floor(3 log n)
        # For n=5, 3*log(5) ~ 4.8, so 4+4=8. Let's use Lambda=8.
        # Sigma (step size) needs to be adapted.
        
        # Initial sigma: 
        # The domain is [-5, 5]^5. The minimizer is in [-4, 4]^5.
        # A good initial sigma is roughly (max - min) / 5 or similar.
        # Let's start with sigma = 2.0.
        
        # We will implement a restart strategy if the population collapses (sigma becomes very small) or stagnates.
        
        max_restarts = 5
        budget_per_restart = budget // max_restarts
        
        for restart_idx in range(max_restarts):
            if eval_count >= budget or time_check():
                break
            
            # Initialize state
            # Start point: 
            # First restart: origin or random? 
            # Since minimizer is in [-4,4], and we don't know where, 
            # maybe start at 0 for the first one, then random for others?
            # Or always random to be safe?
            # Let's use random for all to be robust, but maybe bias the first one to 0?
            # Actually, BBOB functions are often shifted. Random is safer.
            # But if we use CMA-ES, it's global.
            
            if restart_idx == 0:
                x_mean = np.zeros(dim)
            else:
                x_mean = rng.uniform(-4.0, 4.0, dim)
                
            sigma = 2.0
            Lambda = 8 # Population size
            mu = Lambda // 2 # Number of selected individuals
            
            # Weights
            # Standard weights: w_i = log(mu + 0.5) - log(i) for i=1..mu, normalized
            weights = np.zeros(mu)
            for i in range(1, mu + 1):
                weights[i-1] = np.log(mu + 0.5) - np.log(i)
            weights /= np.sum(weights)
            
            # Variance reduction factor
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Strategy parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # State variables
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            hs = np.sqrt((1 - c_c) / (1 + c_c)) + 1
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            # Initial C (identity matrix)
            C = np.eye(dim)
            D = 1.0 # Square root of determinant of C, initially 1
            # Actually, we need the inverse square root of C for sampling.
            # Let's maintain B and D such that C = B * diag(D^2) * B.T
            # Initially B=I, D=1.
            B = np.eye(dim)
            D_vec = np.ones(dim)
            
            # Population history
            pop_x = []
            pop_f = []
            
            # Initial population
            for i in range(Lambda):
                if eval_count >= budget or time_check():
                    break
                # Sample: x = mean + sigma * (B * D * z)
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (B @ (D_vec * z))
                f = func(x)
                eval_count += 1
                pop_x.append(x)
                pop_f.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if eval_count >= budget or time_check():
                break
            
            # CMA-ES Loop
            stagnant_iters = 0
            prev_best_f = best_f
            
            while eval_count < budget and not time_check():
                # Sort population by fitness
                idx = np.argsort(pop_f)
                pop_x = [pop_x[i] for i in idx]
                pop_f = [pop_f[i] for i in idx]
                
                # Update best
                if pop_f[0] < best_f:
                    best_f = pop_f[0]
                    best_x = pop_x[0].copy()
                    stagnant_iters = 0
                else:
                    stagnant_iters += 1
                    
                # Stagnation check: if no improvement for a while, restart
                # Or if sigma is too small
                if stagnant_iters > 30 or sigma < 1e-8:
                    break
                
                # Select top mu
                x_new = np.array(pop_x[:mu])
                
                # Update mean
                x_mean_new = np.sum(weights[:, np.newaxis] * x_new, axis=0)
                
                # Update step size control (ps)
                # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B @ (sqrt(C) * (x_mean_new - x_mean) / sigma)
                # Note: B @ (D_vec * z) was the sampling. The "sqrt(C)" part is B * D_vec.
                # The update term is: sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D_vec * (x_mean_new - x_mean) / sigma)) / sigma ??
                # Standard formula: ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff)/sigma * B * (sqrt(D) * (x_mean_new - x_mean)/sigma)
                # Wait, the sampling was x = mean + sigma * B * D * z.
                # So the "natural gradient" direction is B * D * z.
                # The update for ps uses the weighted mean of the selected z's? No, it uses the difference of means.
                # Let's use the standard CMA-ES update:
                # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B @ (D_vec * (x_mean_new - x_mean) / sigma)
                
                diff = x_mean_new - x_mean
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D_vec * (diff / sigma)))
                
                # Update covariance adaptation
                # C = (1 + c_1 * (1 - c_1) + c_mu) * C + c_1 * (1 - c_1) * (pc * pc.T - (1 - c_c) * C) + c_mu / sigma^2 * sum(weights_i * (y_i * y_i.T))
                # Where y_i = (x_i - x_mean) / sigma
                # We need to update C first, then B and D.
                
                # Update pc
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (diff / sigma)
                
                # Calculate hsig
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (eval_count - Lambda))) < (1.4 + 2 / (dim + 1)) * chi_n)
                
                # Update C
                # We need the outer products of the selected individuals relative to the old mean, scaled by 1/sigma
                # y_i = (x_i - x_mean_old) / sigma
                y = (x_new - x_mean) / sigma
                outer_products = np.sum(weights[:, np.newaxis, np.newaxis] * y[:, np.newaxis, :] * y[:, :, np.newaxis], axis=0)
                
                C = (1 + c_1 * (1 - c_1) + c_mu) * C + c_1 * (1 - c_1) * (pc * pc.T - (1 - c_c) * C) + c_mu * outer_products
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update B and D (Eigen decomposition)
                # This is expensive in Python for small N, but dim=5 is tiny.
                try:
                    D_vec, B = np.linalg.eigh(C)
                    # D_vec contains eigenvalues, we need square roots
                    D_vec = np.sqrt(np.maximum(D_vec, 1e-16))
                    # eigh returns ascending order, we want descending? 
                    # Actually, order doesn't matter for sampling as long as consistent.
                    # But standard CMA-ES usually sorts descending.
                    idx_eig = np.argsort(D_vec)[::-1]
                    D_vec = D_vec[idx_eig]
                    B = B[:, idx_eig]
                except:
                    # If decomposition fails, break
                    break
                    
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update mean
                x_mean = x_mean_new
                
                # Generate new population
                new_pop_x = []
                new_pop_f = []
                for i in range(Lambda):
                    if eval_count >= budget or time_check():
                        break
                    z = rng.standard_normal(dim)
                    x = x_mean + sigma * (B @ (D_vec * z))
                    f = func(x)
                    eval_count += 1
                    new_pop_x.append(x)
                    new_pop_f.append(f)
                    
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                
                if eval_count >= budget or time_check():
                    break
                    
                pop_x = new_pop_x
                pop_f = new_pop_f

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            if eval_count < budget:
                best_f = func(best_x)
                eval_count += 1
            else:
                best_f = np.inf

        return best_x, best_f
