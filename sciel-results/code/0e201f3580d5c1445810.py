"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim))
        lam = 4 + int(3 * np.log(dim)) if dim > 1 else 4
        # Cap population size to ensure we can do multiple generations within budget
        # If budget is 500 and dim is 5, lam=9. 500/9 ~ 55 generations.
        # This is a reasonable number for CMA-ES on 5D problems.
        
        # Initial mean: center of the box
        x_mean = np.zeros(dim)
        
        # Initial step size: heuristic based on box size
        # Box is [-5, 5], width 10. Standard deviation ~ width / (2 * sqrt(dim)) ?
        # Or just a fraction of the box width. 0.5 * width / sqrt(dim) ?
        # Let's start with a moderately large sigma to explore, e.g., 2.0
        sigma = 2.0
        
        # Bounds
        lower = -5.0 * np.ones(dim)
        upper = 5.0 * np.ones(dim)
        
        # CMA-ES weights
        mu = lam // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial state
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        inv_C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Track best
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Evaluate initial point
        f_mean = func(x_mean)
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
        
        eval_count = 1
        rng = np.random.default_rng(42)
        
        # Main loop
        while eval_count < budget and time.time() < deadline:
            # Check if we should stop based on time or convergence
            if sigma < 1e-12:
                break
                
            # Generate population
            pop = np.zeros((lam, dim))
            # Standard normal randoms
            z = rng.standard_normal((lam, dim))
            
            # Transform using Cholesky of C
            # Since C is symmetric positive definite, we can use np.linalg.cholesky
            # But to avoid errors if C becomes singular, we can use eigendecomposition or just update C carefully.
            # For simplicity and robustness in this short run, let's use a simple Cholesky update or just assume C is well-conditioned.
            # A safer way for a short script: compute Cholesky factor.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: reset C to identity if it becomes singular
                C = np.eye(dim)
                L = np.eye(dim)
                
            pop = x_mean + sigma * (z @ L.T)
            
            # Clip to bounds
            np.clip(pop, lower, upper, out=pop)
            
            # Evaluate population
            f_vals = np.zeros(lam)
            for i in range(lam):
                if eval_count >= budget or time.time() >= deadline:
                    break
                f_vals[i] = func(pop[i])
                eval_count += 1
                
                if f_vals[i] < best_f:
                    best_f = f_vals[i]
                    best_x = pop[i].copy()
            
            # If we broke early in evaluation, pad with inf or handle carefully
            # But CMA-ES expects full population. If we ran out of budget, we should exit the loop.
            if eval_count >= budget:
                break
            if time.time() >= deadline:
                break
                
            # Sort by fitness
            order = np.argsort(f_vals)
            y = pop[order]
            f_sorted = f_vals[order]
            
            # Update mean
            x_new = np.sum(weights[:, None] * y[:mu], axis=0)
            
            # Update step size control variables
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * L.T * (x_new - x_mean)
            d = (x_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L.T @ d)
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (eval_count / lam)))) < (1.4 + 2 / (dim + 1)) * chiN
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * d
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu + mu_delta) * C + c_1 * pc * pc.T + c_mu * sum(weights_i * y_i * y_i.T)
            # where y_i are the standardized selected individuals
            
            # Compute delta for correction term
            # This term is often small, we can approximate or include it.
            # For simplicity and speed, let's use the standard update without the small correction term initially, 
            # or include it if we want high fidelity. 
            # c_mu * sum(w_i * v_i v_i^T) where v_i = (y_i - x_mean)/sigma
            
            v = (y[:mu] - x_mean) / sigma
            # Rank-mu update
            rank_mu = np.sum(weights[:, None, None] * (v * v.T), axis=0)
            
            # Rank-one update
            rank_one = np.outer(pc, pc)
            
            # Correction term (mu_delta)
            # mu_delta = c_mu * mueff * (np.sum(weights) - 1)
            # This is usually small. Let's skip it for simplicity in this short run, or include it.
            # Let's include it for correctness.
            mu_delta = c_mu * mueff * (np.sum(weights) - 1)
            
            C = (1 - c_1 - c_mu + mu_delta) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_new
            
            # Update sigma
            sigma *= np.exp(c_sigma / chiN * (np.linalg.norm(ps) / chiN - 1))
            
            # Periodically re-orthogonalize C to prevent numerical drift
            # Every 10 generations or so
            if eval_count % (10 * lam) < lam:
                # Eigendecomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    # Clip negative eigenvalues to 0
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C = (C + C.T) / 2
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    
        return best_x, best_f

# EVOLVE-BLOCK-END
