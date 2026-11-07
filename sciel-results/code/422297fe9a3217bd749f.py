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
        t0 = time.time()
        deadline = t0 + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        d = self.dim
        # Initial population size
        lambda_ = max(4, int(4 + 3 * np.log(d)))
        
        # Initial point and sigma
        x0 = np.zeros(d)
        sigma = 2.0
        
        # Weights
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c1 = (2 / ((d + 1.3) ** 2 + mueff)) / (1 - cc)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        
        # Matrices and vectors
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        # Initial evaluation
        x_mean = x0.copy()
        f_vals = [func(x_mean)]
        best_x = x_mean.copy()
        best_f = f_vals[0]
        
        # Precompute square root of C for efficiency (Cholesky)
        # We will update C and recompute Cholesky each generation
        
        for gen in range(self.budget // lambda_ + 1):
            if time.time() > deadline:
                break
                
            # Generate offspring
            # Compute Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes non-positive definite due to numerical issues
                C = np.eye(d)
                L = np.eye(d)
            
            z = rng.standard_normal((lambda_, d))
            # Transform to normal distribution with covariance C
            y = np.dot(z, L.T)
            x_offspring = x_mean + sigma * y
            
            # Evaluate offspring
            gen_f_vals = []
            for i in range(lambda_):
                # Clip to box [-5, 5]
                x_offspring[i] = np.clip(x_offspring[i], -5, 5)
                f_val = func(x_offspring[i])
                gen_f_vals.append(f_val)
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_offspring[i].copy()
            
            # Sort offspring
            indices = np.argsort(gen_f_vals)
            y_sorted = [x_offspring[i] - x_mean for i in indices]
            f_sorted = [gen_f_vals[i] for i in indices]
            
            # Update mean
            x_mean_new = np.sum([weights[i] * y_sorted[i] for i in range(mu)], axis=0) + x_mean
            x_mean_new = np.clip(x_mean_new, -5, 5)
            
            # Update evolution paths
            # pc update
            pc = (1 - cc) * pc + cc * np.sqrt(mueff) * np.dot(L.T, np.sum([weights[i] * y_sorted[i] for i in range(mu)], axis=0) / sigma)
            
            # ps update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(L.T, np.sum([weights[i] * y_sorted[i] for i in range(mu)], axis=0) / sigma)
            
            # Heuristic for updating C
            delta_hs = (1 - np.exp(-1 / (d + 1))) * np.sqrt((d + 1) / (d - 1))
            
            # Rank-mu update
            v = np.sum([weights[i] * np.outer(y_sorted[i], y_sorted[i]) for i in range(mu)], axis=0)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - cc)**2) * np.eye(d)) + cmu * v
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Add small regularization if needed
            C += 1e-10 * np.eye(d)
            
            # Update sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) <= (1.4 + 2 / (d + 1)) * chiN
            c_sigma = (4 + mueff / d) / (d + 4 + 3 * mueff / d)
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chiN - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            x_mean = x_mean_new
            
            # Check for convergence
            if np.linalg.norm(x_mean - best_x) < 1e-10 and sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
