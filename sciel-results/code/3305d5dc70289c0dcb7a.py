import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        # CMA-ES parameters
        # For dim=5, a small population is efficient.
        # lambda = max(4, int(4 + 3*log(dim))) -> max(4, 4+3*1.6) = 8? 
        # Let's stick to lambda=4 or 6 for faster convergence in low dim.
        # With budget 500, lambda=4 gives ~125 generations.
        # lambda=6 gives ~83 generations.
        # Let's use lambda=4 for more adaptation steps.
        lambda_pop = 4
        mu = lambda_pop // 2  # Selection mass
        
        # Initial state
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-5, 5, self.dim)
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(self.dim)
        
        # Step size control parameters (from CMA-ES literature)
        cmu = 1.0 / (self.dim + 3.0)
        cs = (mu + 2.0) / (self.dim + mu + 5.0)
        ds = 1.0 + 2.0 * max(0, (mu - 1.0) / (self.dim + 1.0)) + cs
        
        # Path lengths
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        # Initial evaluations
        evals = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        evals += 1
        
        # Pre-compute weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Pre-compute sqrt(C) for sampling? No, C changes.
        # We need to update C and sample from N(x_mean, sigma^2 * C)
        
        for gen in range(self.budget // lambda_pop + 1):
            if time.time() > deadline:
                break
            if evals >= self.budget:
                break
                
            # Generate offspring
            offspring = []
            f_vals = []
            
            # Update C and sigma before sampling for this generation? 
            # Standard CMA-ES updates after selection.
            # For the first generation, C=I, sigma=2.0.
            
            # Compute inverse square root of C for sampling? 
            # Actually, standard CMA-ES samples y ~ N(0, C) and x = x_mean + sigma * y.
            # We can use Cholesky decomposition or eigendecomposition.
            # For dim=5, eigendecomposition is fast and stable.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            B = eigvecs @ np.diag(np.sqrt(eigvals))
            
            for i in range(lambda_pop):
                if evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Sample y ~ N(0, I)
                y = rng.standard_normal(self.dim)
                # Transform to x
                x = x_mean + sigma * (B @ y)
                # Clip to bounds
                x = np.clip(x, -5, 5)
                
                f = func(x)
                evals += 1
                
                offspring.append(x)
                f_vals.append(f)
                
                # Update global best
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(offspring) < lambda_pop:
                break
            
            # Sort offspring by fitness
            indices = np.argsort(f_vals)
            selected_x = [offspring[i] for i in indices[:mu]]
            selected_f = [f_vals[i] for i in indices[:mu]]
            
            # Update x_mean
            x_new_mean = np.sum(np.array(selected_x) * weights[:, None], axis=0)
            
            # Update step size (sigma)
            # d_sigma correction
            d_sigma = 1.0 + 2.0 * max(0, (mueff - 1.0) / (self.dim + mueff - 1.0)) + cs
            
            # Old sigma for path update
            sigma_old = sigma
            
            # Update path ps
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (B.T @ (x_new_mean - x_mean) / sigma_old)
            
            # Update h_sigma
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs)**(2.0 * (evals // lambda_pop + 1)))) < (1.4 + 2.0 / (self.dim + 1.0)) * np.sqrt((1.0 - (1.0 - cs)**(2.0 * (evals // lambda_pop + 1))))
            
            # Update pc
            pc = (1.0 - cs) * pc + h_sigma * np.sqrt(cs * (2.0 - cs) * mueff) * (x_new_mean - x_mean) / sigma_old
            
            # Update C
            # Delta h term
            delta_h = (1.0 - h_sigma) * cs * (2.0 - cs)
            
            # Rank-one and rank-mu update
            C = (1.0 + cmu * (mueff - 1.0) - cmu * delta_h) * C
            C += cmu * delta_h * np.outer(pc, pc)
            C += cmu * (np.sum(weights**2, axis=0) * np.eye(self.dim) - 2.0 * np.sum(weights[:, None] * np.outer(np.array(selected_x) - x_mean, np.array(selected_x) - x_mean), axis=0) / sigma_old**2)
            
            # Symmetrize C
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma = sigma_old * np.exp(cs / d_sigma * (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs)**(2.0 * (evals // lambda_pop + 1))) - 1.0))
            
            # Update x_mean
            x_mean = x_new_mean
            
            # Optional: Restart if sigma is too small or too large
            if sigma < 1e-12 or sigma > 1e6:
                break

        return best_x, best_f
