"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(n)) if n > 1 else 4
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - mueff / (n + 1.7) + c_sigma) / ((n + 1.3) ** 2 + mueff)
        c_mu = (2 * min(1, (mueff - 1 + w[0]) / (n + 2)) - c_1) / \
               ((n + 1.3) ** 2 + mueff)
        
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Initial state
        # Start near origin as minimizer is in [-4, 4]^n
        x_mean = np.zeros(n)
        # Initial sigma: cover the box roughly. Box width is 10.
        # Standard heuristic: sigma0 = 0.3 * range or similar. 
        # For Bent Cigar, we want to explore broadly initially.
        sigma = 1.0 
        
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        def inv_sqrtm(A):
            """Compute inverse square root of symmetric matrix A."""
            # Use eigen decomposition for stability
            try:
                eigvals, eigvecs = np.linalg.eigh(A)
                # Clip small negative eigenvalues due to numerical noise
                eigvals = np.clip(eigvals, 1e-12, None)
                inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
                return (eigvecs * inv_sqrt_eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                # Fallback to identity if it fails
                return np.eye(n)
        
        C_inv_sqrt = inv_sqrtm(C)
        
        best_x = x_mean.copy()
        best_f = np.inf
        evaluations = 0
        
        # Initial evaluation
        f_vals = []
        x_pop = []
        
        # Generate initial population
        for i in range(lam):
            z = np.random.randn(n)
            x = x_mean + sigma * (C_inv_sqrt @ z)
            x = np.clip(x, -5, 5)
            x_pop.append(x)
            
        for x in x_pop:
            f = func(x)
            f_vals.append(f)
            evaluations += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Sort population
        idx = np.argsort(f_vals)
        x_pop = [x_pop[i] for i in idx]
        f_vals = [f_vals[i] for i in idx]
        
        # Main loop
        while evaluations < budget:
            if time.time() > deadline:
                break
                
            # Update mean
            x_new_mean = sum(w[i] * x_pop[i] for i in range(mu))
            
            # Update evolution paths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (
                C_inv_sqrt @ (x_new_mean - x_mean) / sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                (C_inv_sqrt @ (x_new_mean - x_mean)) / sigma)
            
            # Update covariance matrix
            delta_sig = (np.linalg.norm(ps) / chi_n - 1) * c_sigma * d_sigma
            sigma *= np.exp(delta_sig)
            
            v = (ps + c_sigma * (1 - c_sigma)) * ps / sigma
            C = (1 - c_1 - c_mu + c_1 * c_sigma * (2 - c_sigma) + c_mu) * C + \
                c_1 * (np.outer(pc, pc) - (1 - c_c) * C) + \
                c_mu * sum(w[i] * np.outer(x_pop[i] - x_mean, x_pop[i] - x_mean) for i in range(mu))
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            C += 1e-12 * np.eye(n)
            
            x_mean = x_new_mean
            C_inv_sqrt = inv_sqrtm(C)
            
            # Generate next population
            x_pop = []
            f_vals = []
            for i in range(lam):
                z = np.random.randn(n)
                x = x_mean + sigma * (C_inv_sqrt @ z)
                x = np.clip(x, -5, 5)
                x_pop.append(x)
                
            for x in x_pop:
                f = func(x)
                f_vals.append(f)
                evaluations += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    
                if time.time() > deadline:
                    break
                    
            # Sort population for next iteration
            if len(f_vals) == lam:
                idx = np.argsort(f_vals)
                x_pop = [x_pop[i] for i in idx]
                f_vals = [f_vals[i] for i in idx]
            else:
                # If we broke early due to time, we might not have a full sorted pop
                # But the loop condition checks evaluations < budget.
                # If we broke inner loop, we should probably break outer too or handle carefully.
                # Given the strict time limit, breaking outer is safer.
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
