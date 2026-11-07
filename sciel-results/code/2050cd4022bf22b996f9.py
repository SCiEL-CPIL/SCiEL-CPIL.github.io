import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 280  # Safety margin
            
            # CMA-ES Parameters
            n = self.dim
            lam = 4 + int(3 * np.log(n))  # ~9 for n=5
            c_sigma = 1.0 / (4.0 * np.sqrt(n))
            c_c = 4.0 / (n + 4.0)
            c_mu = 1.0 / (n + 13.0)
            c_chi = 1.0 / (n + 25.0)
            
            # Weights for recombination
            weights = np.log(lam / 2.0 + 0.5) - np.log(np.arange(1, lam + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights[:lam//2 + 1]**2)
            
            # Initial state
            rng = np.random.default_rng(42)
            x_mean = rng.uniform(-5, 5, n)
            sigma = 2.0
            C = np.eye(n)
            C_inv_sqrt = np.eye(n) # Will update
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            calls_made = 0
            best_x = x_mean.copy()
            best_f = np.inf
            
            def count_func(x):
                nonlocal calls_made
                calls_made += 1
                return func(x)
                
            # Evaluate initial point
            f_mean = count_func(x_mean)
            best_f = f_mean
            
            while calls_made < self.budget:
                # Check time
                if time.time() > deadline:
                    break
                
                # Generate population
                ys = []
                fs = []
                for i in range(lam):
                    # Sample from N(0, C)
                    # Use Cholesky or SVD for sqrt(C)
                    # For efficiency, we can use a fixed transformation if C changes slowly, 
                    # but here we update C. 
                    # Simple approximation: use eigen decomposition or just random normal * sigma if C is identity initially.
                    # Proper CMA-ES requires transforming standard normal by C^{1/2}.
                    
                    # To keep it simple and robust without heavy linear algebra overhead per sample:
                    # We can use the fact that for small n, computing C^{1/2} is fast.
                    # However, doing it inside the loop for each sample is wasteful.
                    # Let's compute C^{1/2} once per generation.
                    
                    break # Placeholder, need to restructure loop
                    
                # Restructure: Compute C^{1/2} once
                # Eigendecomposition of C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
                
                ys = []
                fs = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = count_func(y)
                    ys.append(y)
                    fs.append(fy)
                    
                    if calls_made >= self.budget:
                        break
                
                if not ys:
                    break
                    
                # Sort
                idx = np.argsort(fs)
                ys = np.array(ys)[idx]
                fs = np.array(fs)[idx]
                
                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = ys[0].copy()
                    
                # Update mean
                mu = lam // 2 + 1 # Number of parents
                x_new_mean = sum(weights[i] * ys[i] for i in range(mu))
                
                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new_mean - x_mean) / sigma
                
                # Hessianian norm
                norm_ps = np.linalg.norm(p_sigma)
                chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
                
                # Update sigma
                if norm_ps / chi_n < 1.0:
                    sigma *= np.exp(c_sigma / c_chi * (norm_ps / chi_n - 1))
                    
                # Update p_c
                h_sigma = (norm_ps / (1 + 5*c_sigma) * chi_n) > (1.4 + 2/(n+1))
                h_sigma = float(h_sigma)
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x_new_mean - x_mean) / sigma
                
                # Update C
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(p_c, p_c)
                rank_mu = sum(weights[i] * np.outer(ys[i] - x_mean, ys[i] - x_mean) for i in range(mu))
                
                C = (1 + c_sigma * c_chi * (norm_ps**2 - n) + delta_h - c_mu) * C + c_sigma * rank_one + c_mu * rank_mu
                
                # Ensure C is positive definite and symmetric
                C = (C + C.T) / 2
                # Add small regularization if needed
                C += 1e-12 * np.eye(n)
                
                x_mean = x_new_mean
                
                # Stagnation check: If sigma is very small or no improvement in last 10 generations
                # Simple check: if sigma < 1e-4, restart
                if sigma < 1e-4:
                    # Restart around best_x with larger sigma
                    x_mean = best_x + rng.uniform(-1, 1, n)
                    x_mean = np.clip(x_mean, -5, 5)
                    sigma = 1.0
                    C = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    
            return best_x, best_f
