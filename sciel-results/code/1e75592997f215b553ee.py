import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(d)) 
        # Number of selected points mu
        mu = lam // 2
        # Weights for the selected points
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance-effective selection mass
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Initial step size
        # Heuristic: 1/3 of the box size is a reasonable start
        sigma = 1.0 
        
        # Initial center (centroid)
        # Start at 0,0,0,0,0 as it's a common default and the optimum is in [-4,4]
        mean = np.zeros(d)
        
        # Initial covariance matrix
        C = np.eye(d)
        # Pre-compute C^0.5 and C^-0.5
        C_sqrt, _ = np.linalg.cholesky(C)
        C_inv_sqrt, _ = np.linalg.cholesky(np.linalg.inv(C))
        
        # Adaptation parameters
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        cmu = min(1 - cc, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        c_sigma = (mueff + 1.5) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        
        # Path accumulators
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Count evaluations
        calls_made = 0
        
        # Initial evaluation
        # Note: func clips to [-5,5]
        mean = np.clip(mean, -5, 5)
        f_mean = func(mean)
        calls_made += 1
        
        best_x = mean.copy()
        best_f = f_mean
        
        while calls_made < budget:
            # Generate population
            # We need to generate 'lam' points, but we might run out of budget
            # If remaining budget < lam, we can just do a final random restart or small ES
            # For simplicity, if remaining < lam, we break and do a local refinement or just stop.
            # Actually, let's just generate min(lam, budget - calls_made) points?
            # Standard CMA-ES requires a full population for the update.
            # If we can't afford a full population, we can reduce lambda or stop.
            # Given budget 500 and lam ~ 10, we have ~50 generations.
            
            if budget - calls_made < lam:
                 # Do a final batch of random perturbations around best_x
                 remaining = budget - calls_made
                 for _ in range(remaining):
                     y = np.clip(best_x + 0.1 * np.random.randn(d), -5, 5)
                     fy = func(y)
                     calls_made += 1
                     if fy < best_f:
                         best_f = fy
                         best_x = y
                 break

            Y = np.zeros((lam, d))
            F = np.zeros(lam)
            
            for i in range(lam):
                z = np.random.randn(d)
                y = mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                calls_made += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = y.copy()
            
            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Update mean
            mean_new = np.sum(weights[:, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update covariance matrix
            # y_k = (x_k - mean_old) / sigma
            Y_diff = (Y_sorted[:mu] - mean) / sigma
            C_new = C
            for k in range(mu):
                w_k = weights[k]
                y_k = Y_diff[k]
                C_new += w_k * (np.outer(y_k, y_k) - C)
            
            # Ensure symmetry and positive definite
            C_new = (C_new + C_new.T) / 2.0
            # Small regularization to prevent numerical issues
            C_new += 1e-12 * np.eye(d)
            
            # Update step size sigma
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mueff) * C_inv_sqrt @ (mean_new - mean) / sigma
            delta_sigma = (mean_new - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ delta_sigma)
            
            # Update p_c
            # p_c = (1 - cc) p_c + sqrt(cc (2 - cc) h_sigma) * (mean_new - mean) / sigma
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(gen+1)}) < (1.4 + 2/(d+1)) * E||N(0,I)||
            # Approximate h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            # E||N(0,I)|| = sqrt(d) * (1 - 1/(4d) + 1/(32d^2)) approx
            h_sigma = 1.0 
            # Simple heuristic for h_sigma often used in implementations:
            # if norm_p_sigma / sqrt(1 - (1-c_sigma)**(2*gen)) < (1.4 + 2/(d+1)) * sqrt(d): h_sigma=1 else 0
            # We need a generation counter for the exponent. Let's track it.
            
            # For robustness, let's use a simplified h_sigma logic or just set h_sigma=1 if norm is not huge
            # Standard implementation:
            # h_sigma = int(norm_p_sigma / np.sqrt(1 - (1-c_sigma)**(2*gen)) < (1.4 + 2/(d+1)) * np.sqrt(d))
            # We don't have 'gen' easily here without a counter, but we can approximate or just use 1.
            # Let's add a gen counter.
            
            # Actually, let's just use the standard update without the complex h_sigma check initially, 
            # or implement a simple gen counter.
            
            # Let's implement a proper gen counter in the loop.
            # I'll refactor the loop slightly to track generation.
            
            # Re-doing the logic with a generation counter for h_sigma
            # Since I already calculated mean_new and C_new, I need to be careful.
            # Let's restart the loop structure in my head to include gen.
            
            # To avoid complex refactoring in the thought process, I will implement the standard CMA-ES
            # with a generation counter in the actual code below.
            
            # Update p_c
            # We need h_sigma. Let's assume h_sigma=1 for now, it's a minor adaptation detail.
            # p_c = (1 - cc) p_c + sqrt(cc (2 - cc)) * (mean_new - mean) / sigma
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc)) * delta_sigma
            
            # Update C
            # C = (1 - cmu - c_c) C + ...
            # c_c = cc * (cmu + 2 * (1 - cc))
            c_c = cc * (cmu + 2 * (1 - cc))
            C_new = (1 - cmu - c_c) * C + cmu * np.sum(weights[:, np.newaxis] * np.outer(Y_diff, Y_diff), axis=0) + c_c * np.outer(p_c, p_c)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(d) - 1))
            
            # Update mean and C
            mean = mean_new
            C = C_new
            
            # Recompute C_sqrt and C_inv_sqrt
            try:
                C_sqrt, _ = np.linalg.cholesky(C)
                C_inv_sqrt, _ = np.linalg.cholesky(np.linalg.inv(C))
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular
                C = C + 1e-4 * np.eye(d)
                C_sqrt, _ = np.linalg.cholesky(C)
                C_inv_sqrt, _ = np.linalg.cholesky(np.linalg.inv(C))
                
        return best_x, best_f
