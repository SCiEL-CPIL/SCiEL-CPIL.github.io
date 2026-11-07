"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """CMA-ES-like optimizer using scipy's differential_evolution or a custom CMA implementation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Use a deterministic seed for reproducibility
        seed = 42
        rng = np.random.default_rng(seed)
        
        # Initial population for CMA-ES
        dim = self.dim
        budget = self.budget
        
        # Start with a reasonable initial point
        x0 = np.zeros(dim)
        # Try a few random starting points to pick the best
        candidates = [x0]
        for _ in range(10):
            candidates.append(rng.uniform(-5, 5, dim))
        
        best_x = candidates[0]
        best_f = func(best_x)
        for c in candidates[1:]:
            fc = func(c)
            if fc < best_f:
                best_x = c
                best_f = fc
        
        # Use CMA-ES via scipy if available, otherwise fall back to a simple CMA-ES
        # Let's implement a basic CMA-ES from scratch for better control
        
        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))  # lambda
        pop_size = min(pop_size, budget - 20)
        sigma = 1.0
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + 4 + dim / (dim + 4))
        cmu = min(2 - cc + 2 / (dim + 1), 0.5)
        
        x_mean = best_x.copy()
        f_mean = best_f
        
        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for sqrt
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-12))
        
        cs = (4 + dim / (dim + 4)) / (dim + 4.1)
        ds = 1 + 2 * max(0, (cc - 1) / (dim + 1)) + cmu
        
        # Cumulation paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        calls_used = 11  # 1 + 10 initial evaluations
        remaining = budget - calls_used
        
        if remaining <= 0:
            return best_x, best_f
        
        # Adaptive sigma update
        sigma_update = 1.0
        target_sigma = sigma
        
        for gen in range(remaining // pop_size):
            if calls_used >= budget:
                break
            
            # Sample population
            offspring = []
            for i in range(pop_size):
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                x_i = np.clip(x_mean + sigma * y, -5, 5)
                offspring.append((x_i, None))
            
            # Evaluate offspring
            for i, (x_i, _) in enumerate(offspring):
                if calls_used >= budget:
                    break
                f_i = func(x_i)
                offspring[i] = (x_i, f_i)
                calls_used += 1
            
            # Sort offspring by fitness
            offspring = [o for o in offspring if o[1] is not None]
            offspring.sort(key=lambda x: x[1])
            
            # Update mean
            mu = min(pop_size // 2, len(offspring))
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            x_new_mean = np.sum([weights[i] * offspring[i][0] for i in range(mu)], axis=0)
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(x_new_mean - x_mean) / (sigma * np.sqrt(mu_eff)) - 1))
            sigma = np.clip(sigma, 1e-8, 100)
            
            # Update covariance matrix
            delta_mean = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * delta_mean
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))) else 0.0
            
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * delta_mean
            
            C = (1 + cmu * (h_sigma * (2 - cc) - cc)) * C + cmu * np.outer(delta_mean, delta_mean) - cmu * (1 - h_sigma * (2 - cc)) * C
            
            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update mean
            x_mean = x_new_mean
            f_mean = offspring[0][1]
            
            # Update best
            if offspring[0][1] < best_f:
                best_x = offspring[0][0].copy()
                best_f = offspring[0][1]
            
            # Re-decompose C for next iteration
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            
            if calls_used >= budget:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
