"""CMA-ES optimizer for BBOB f13 Sharp Ridge."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # lambda: population size. For dim=5, lambda ~ 4 + floor(3 log(5)) = 4+5 = 9. 
        # Let's use lambda = 10 for robustness.
        lam = 10
        
        # Initial sigma: 2.0 is reasonable for [-5,5] box, but f13 is sharp.
        # Start a bit smaller to avoid overshooting the ridge immediately?
        # Standard CMA-ES init sigma is 1.0 or 2.0. Let's try 1.0.
        sigma = 1.0
        
        # Initial mean: center of box
        x_mean = np.zeros(dim)
        
        # Covariance matrix: identity
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Eigendecomposition for sampling (initially identity)
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Weights
        mu = lam // 2
        w = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Best so far
        x_best = x_mean.copy()
        f_best = float('inf')
        
        # Pre-allocate storage for evaluations in this iteration
        # We will evaluate lam points per generation.
        # Total budget is 500. 
        # Generations approx 50.
        
        call_count = 0
        
        # Helper to sample from N(0, C)
        def sample_cma():
            z = np.random.standard_normal(dim)
            return B @ (D * z)
            
        def update_eigen():
            nonlocal C, C_inv, D, B
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive definite
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs
            C_inv = (eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T)
            
        # Initial evaluation
        f_mean = func(x_mean)
        call_count += 1
        f_best = f_mean
        x_best = x_mean.copy()
        
        while call_count < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate population
            x_pop = []
            for i in range(lam):
                # Sample
                z = np.random.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                # Clip to box
                y = np.clip(y, -5, 5)
                x_pop.append(y)
                
            # Evaluate population
            f_pop = []
            for i in range(lam):
                if call_count >= budget:
                    break
                fi = func(x_pop[i])
                f_pop.append(fi)
                call_count += 1
                
                # Update best
                if fi < f_best:
                    f_best = fi
                    x_best = x_pop[i].copy()
                    
                # Check time mid-evaluation
                if time.time() > deadline:
                    # Finish evaluating current batch if possible? 
                    # Or break out? 
                    # Breaking out might leave f_pop incomplete.
                    # Let's just stop the loop if we hit deadline.
                    # We need to handle the case where f_pop is shorter than lam.
                    pass
            
            # If we ran out of budget or time, break
            if call_count >= budget or time.time() > deadline:
                break
            
            # Sort population by fitness
            # Note: f_pop might be shorter if we broke early due to budget/time
            # But we check budget/time before appending. 
            # If we broke *during* the for loop, f_pop has len < lam.
            # CMA-ES requires full population for standard update.
            # To be safe, if we don't have full population, we stop.
            if len(f_pop) < lam:
                break
                
            idx = np.argsort(f_pop)
            x_sorted = [x_pop[i] for i in idx]
            f_sorted = [f_pop[i] for i in idx]
            
            # Weighted mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += w[i] * x_sorted[i]
                
            # Update step size control
            # p_sigma
            x_norm = x_new_mean - x_mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D @ (C_inv @ x_norm))) / sigma
            
            # h_sigma condition
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2) + 1 / (8 * dim**4))
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (call_count - 1))) < (1.4 + 2 / (dim + 1)) * chi_n)
            
            # p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (D @ x_norm)) / sigma
            
            # Update covariance
            delta_h = (2 - h_sigma) * c_c - 1
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C
            for i in range(mu):
                y_i = x_sorted[i] - x_mean
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Ensure sigma is reasonable
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            # Update mean
            x_mean = x_new_mean
            
            # Periodic eigen update (every 10 iterations or so)
            # Or every iteration for dim=5? It's cheap.
            if call_count % 10 == 0 or call_count < 20:
                update_eigen()
                
        return x_best, f_best
# EVOLVE-BLOCK-END
