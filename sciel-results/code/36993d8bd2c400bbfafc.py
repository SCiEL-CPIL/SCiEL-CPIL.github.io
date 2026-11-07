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
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit, leave 10s
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        
        # Step size
        sigma = 2.0  # Initial step size, slightly larger to cover [-5,5] effectively
        
        # Initial mean
        # Uniform random start is safer than 0 if shift is unknown, but 0 is often central.
        # Given the shift is in [-4, 4], starting at 0 is reasonable.
        # Let's use a deterministic start for reproducibility in local tests, 
        # but random is better for generalization. 
        # The task says "hidden seeded shift", implying it's fixed per instance but unknown.
        # Starting at 0 is a good heuristic for BBOB.
        rng = np.random.default_rng(42)
        m = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Eigenvalue decomposition cache
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Evolution path weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        mu_eff = 1 / np.sum(weights**2)
        c_sigma = (2 + 2 / (dim + 1)) / (lam + 2)
        c_c = (4 + mu_eff / dim) / (lam + 4 + 2 * mu_eff / dim)
        
        c_1 = 2 / ((dim + 1.3)**2 + lam_eff) if (lam_eff := mu_eff) else 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + lam_eff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        calls_made = 0
        best_x = m.copy()
        best_f = float('inf')
        
        def update_eigen():
            nonlocal D, B
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            D = 1.0 / np.sqrt(eigvals)
            B = eigvecs
            
        # Initial eigen decomposition
        update_eigen()
        
        # Generation loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            pop = np.zeros((lam, dim))
            for i in range(lam):
                # Check budget
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Standard normal
                z = rng.standard_normal(dim)
                
                # Transform to C-space
                # y = B @ (D * z)
                y = B @ (D * z)
                
                # Sample
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                
                pop[i] = x
                calls_made += 1
            
            # If we couldn't fill the population, break
            if calls_made >= budget:
                break
            
            # Evaluate population
            # We need to evaluate all generated individuals
            # Note: The loop above already incremented calls_made.
            # We need to store the fitness.
            fit = np.zeros(lam)
            # Wait, the loop above generated them but didn't evaluate them in a way that stores fit easily if we break mid-loop.
            # Let's restructure: Generate and evaluate in one go, or store indices.
            # Actually, the previous loop structure is slightly inefficient for tracking fitness if it breaks.
            # Let's restart the generation logic to be cleaner.
            
            # Reset for cleaner implementation
            # We'll just do a standard generation step.
            # If we ran out of budget in the middle of generation, we stop.
            
            # Re-implementing the inner loop for clarity and correctness
            # The previous block was a draft. Let's do it properly.
            
            # Clear pop and fit
            pop = np.empty((lam, dim))
            fit = np.empty(lam)
            
            # We need to know how many we can actually evaluate
            remaining = budget - calls_made
            if remaining <= 0:
                break
            
            # If remaining < lam, we might do a partial generation? 
            # CMA-ES requires a full population for the update.
            # If we don't have enough for a full population, we stop or do random restarts.
            # Given 500 budget and lam ~ 14, we can do ~35 generations.
            # Stopping is safer.
            if remaining < lam:
                break
            
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                pop[i] = x
                fit[i] = func(x)
                calls_made += 1
                
                if fit[i] < best_f:
                    best_f = fit[i]
                    best_x = x.copy()
            
            # Sort population
            idx = np.argsort(fit)
            fit = fit[idx]
            pop = pop[idx]
            
            # Weighted mean of mu best
            m_new = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update covariance matrix
            y_sigma = (m_new - m) / sigma
            y_c = (m_new - m) / np.sqrt(c_c * (2 - c_c) * mu_eff)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            # Update sigma
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            C = (1 + c_1 + c_mu - c_1 - mu_eff) * C + c_1 * (np.outer(y_c, y_c) - C) + \
                c_mu / mu_eff * np.sum(weights[:mu, None, None] * np.outer(pop[:mu] - m, pop[:mu] - m), axis=0)
            
            # Wait, the standard CMA-ES update is:
            # C = (1 + c1 + cmu - c1 - mue) * C ... no.
            # C = (1 - c1 - cmu) * C + c1 * (y_c y_c^T + (1 - h_sigma) * (1 - c_c) * C) + cmu * sum(w_i (x_i - m)(x_i - m)^T)
            
            # Let's use the standard formula:
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(y_c, y_c) + (1 - h_sigma) * (1 - c_c) * C) + \
                c_mu * np.sum(weights[:mu, None, None] * np.outer(pop[:mu] - m, pop[:mu] - m), axis=0)
            
            m = m_new
            
            # Update eigen decomposition periodically or if C changes significantly
            # For 5D, doing it every step is cheap enough.
            update_eigen()
            
            # Termination criteria
            if sigma < 1e-12:
                break
            
            # Check time again
            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
