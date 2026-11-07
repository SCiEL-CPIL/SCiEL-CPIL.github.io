import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x_mean, sigma, C, evals_left, deadline, best_x, best_f):
        """Run CMA-ES and return updated state."""
        n = self.dim
        lambda_ = 4 + int(3 * np.log(n))
        
        # Control parameters
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Pre-compute Cholesky
        try:
            B = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            B = np.eye(n)
            
        E_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        no_improve = 0
        max_no_improve = 30
        
        while evals_left > 0:
            # Time guard
            if time.time() > deadline:
                break
                
            # Check for stagnation - restart with larger sigma
            if no_improve > max_no_improve:
                # Perturb mean and increase sigma
                x_mean = x_mean + 2.0 * np.random.randn(n)
                sigma = max(sigma * 2.0, 1.0)
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    B = np.eye(n)
                no_improve = 0
                continue
            
            # Generate population
            z = np.random.randn(lambda_, n)
            y = sigma * (z @ B.T)
            x = x_mean[None, :] + y
            
            # Evaluate fitness
            if evals_left < lambda_:
                for i in range(evals_left):
                    f = func(x[i])
                    if f < best_f:
                        best_f = f
                        best_x = x[i].copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                    evals_left -= 1
                break
            
            f = np.array([func(x[i]) for i in range(lambda_)])
            evals_left -= lambda_
            
            # Sort by fitness
            idx = np.argsort(f)
            x_sorted = x[idx]
            f_sorted = f[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Recombination
            x_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update step size
            # Compute C^{-1/2} using eigen decomposition for small n
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            d_sigma_vec = C_inv_sqrt @ ((x_new - x_mean) / sigma)
            
            # Proper h_sigma calculation
            h_sigma = float(np.linalg.norm(p_sigma) / np.sqrt(1 - (1-c_sigma)**2) < 
                           (1 + 1.5/n) * E_norm)
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_sigma_vec
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * ((x_new - x_mean) / sigma)
            
            # Update C
            y_sorted = (x_sorted[:mu] - x_mean[None, :]) / sigma
            C_mu = np.sum(weights[:, None, None] * y_sorted[:, :, None] * y_sorted[:, None, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * C_mu
            
            # Update mean
            x_mean = x_new
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E_norm - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            
            # Recompute Cholesky for next iteration
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                sigma = max(sigma, 1.0)
        
        return x_mean, sigma, C, evals_left, best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = float('inf')
        
        # Multi-start strategy: try different initializations
        n_starts = 3
        budget_per_start = budget // n_starts
        remaining = budget
        
        for start_idx in range(n_starts):
            if remaining <= 0 or time.time() > deadline:
                break
            
            # Different starting points for diversity
            if start_idx == 0:
                x_mean = np.zeros(n)
                sigma = 2.0
            elif start_idx == 1:
                x_mean = np.random.uniform(-3, 3, n)
                sigma = 1.5
            else:
                x_mean = np.random.uniform(-4, 4, n)
                sigma = 1.0
            
            C = np.eye(n)
            evals_left = min(budget_per_start, remaining)
            
            x_mean, sigma, C, evals_left, best_x, best_f = self._cma_es(
                func, x_mean, sigma, C, evals_left, deadline, best_x, best_f
            )
            remaining -= (budget_per_start - evals_left)
        
        # Use remaining budget for final refinement around best point
        if remaining > 0 and time.time() < deadline:
            x_mean = best_x.copy()
            sigma = 0.5
            C = np.eye(n)
            evals_left = remaining
            x_mean, sigma, C, evals_left, best_x, best_f = self._cma_es(
                func, x_mean, sigma, C, evals_left, deadline, best_x, best_f
            )
        
        return best_x, best_f
