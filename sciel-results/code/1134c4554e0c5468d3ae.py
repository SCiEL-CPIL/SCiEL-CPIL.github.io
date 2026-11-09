"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4 and cumulative step-size control."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        lam = 4  # population size
        
        # CMA-ES parameters
        mu = lam // 2  # number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        x = np.zeros(n)
        fx = func(x)
        calls_made = 1
        deadline = time.time() + 290.0  # 300s limit, 10s safety margin
        
        B = np.eye(n)
        D = 1.0
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        sigma = 1.0
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            y_list = []
            for i in range(lam):
                y = x + sigma * (B @ (D * rng.standard_normal(n)))
                y = np.clip(y, -5, 5)
                y_list.append(y)
            
            # Evaluate population
            f_list = []
            for y in y_list:
                fy = func(y)
                f_list.append(fy)
                calls_made += 1
                if calls_made >= self.budget:
                    break
            
            if calls_made >= self.budget:
                break
            
            # Sort by fitness
            idx = np.argsort(f_list)
            y_sorted = [y_list[i] for i in idx]
            f_sorted = [f_list[i] for i in idx]
            
            # Update best solution
            if f_sorted[0] < fx:
                x, fx = y_sorted[0], f_sorted[0]
            
            # Update mean
            y_mean = sum(weights[i] * y_sorted[i] for i in range(mu))
            
            # Update covariance matrix adaptation
            y_diff = y_mean - x
            x = y_mean
            
            # Update step-size control (cumulative step-size adaptation)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (sigma) * (B @ (D * y_diff))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))
            sigma = np.clip(sigma, 1e-10, 100)
            
            # Update evolution path for covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_diff
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (n + 1)) * np.sqrt(n) else 0
            
            # Update B and D (simplified: use diagonal approximation)
            # For simplicity, we keep B and D fixed but could update them
            # In a full implementation, we would update the covariance matrix C
            # and do eigendecomposition. Here we use a simplified approach.
            
            # Simple covariance update (diagonal only for speed)
            C = np.diag(np.array([np.mean([(y_sorted[i] - x)[j]**2 for i in range(mu)]) for j in range(n)]))
            # Update D and B via eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-15)
            D = np.sqrt(eigenvalues)
            B = eigenvectors
        
        return x, fx
# EVOLVE-BLOCK-END
