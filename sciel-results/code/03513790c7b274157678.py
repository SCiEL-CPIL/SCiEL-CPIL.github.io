"""Rewritten optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Parameters
        # lambda: population size
        lam = 4 + int(3 * np.log(n))
        # mu: number of best points for update
        mu = lam // 2
        # weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # CMA-ES parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        # Start near center of box
        x_mean = np.zeros(n)
        sigma = 2.0  # Initial step size, box is [-5, 5], so 2 is reasonable
        
        # Covariance matrix
        C = np.eye(n)
        # Precompute inverse square root of C for sampling
        # Initially C is identity, so invsqrt_C is identity
        invsqrt_C = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Calls counter
        calls = 0
        best_fx = float('inf')
        best_x = x_mean.copy()
        
        # Main loop
        while calls < budget:
            # Generate population
            pop_x = np.zeros((lam, n))
            pop_f = np.zeros(lam)
            
            for i in range(lam):
                # Sample from N(0, C)
                # z ~ N(0, I)
                z = np.random.randn(n)
                # x = x_mean + sigma * invsqrt_C @ z
                x = x_mean + sigma * (invsqrt_C @ z)
                # Clip to bounds? The function clips, but we can also clip here to avoid wasted calls if we want.
                # However, CMA-ES assumes the function is defined everywhere. The task says func clips.
                # So we can just pass x.
                pop_x[i] = x
                pop_f[i] = func(x)
                calls += 1
                
                if pop_f[i] < best_fx:
                    best_fx = pop_f[i]
                    best_x = x.copy()
                    
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            # Sort by fitness
            idx = np.argsort(pop_f)
            pop_x = pop_x[idx]
            pop_f = pop_f[idx]
            
            # Update mean
            x_new_mean = np.sum(weights[:, np.newaxis] * pop_x[:mu], axis=0)
            
            # Update evolution path for sigma
            # Need inverse square root of C
            # We update C later, so we use current invsqrt_C
            # y = invsqrt_C @ (x_new_mean - x_mean) / sigma
            y = invsqrt_C @ (x_new_mean - x_mean) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            
            # Hessian update for sigma
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // lam))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))) else 0
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update covariance matrix
            # p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_new_mean - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_new_mean - x_mean) / sigma
            
            # Delta H correction
            delta_h = (1 - h_sigma) * c_c * (1 - c_c) * np.dot(p_c, p_c) / n
            
            # Update C
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * (np.outer(p_c, p_c) - delta_h * C)
            for i in range(mu):
                diff = (pop_x[i] - x_mean) / sigma
                C += c_mu * weights[i] * (np.outer(diff, diff) - C)
            
            # Regularize C to ensure positive definiteness
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues
            eigvals = np.clip(eigvals, 1e-12, 1e12)
            # Reconstruct C
            C = (eigvecs * eigvals) @ eigvecs.T
            
            # Update inverse square root
            eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
            invsqrt_C = eigvecs @ (eigvecs.T * eigvals_inv_sqrt)
            
            x_mean = x_new_mean
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
