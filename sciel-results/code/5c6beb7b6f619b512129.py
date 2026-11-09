import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Initial mean
        m = np.zeros(dim)
        # Initial sigma
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (for sampling)
        # We can just sample standard normal and transform? 
        # For dim=5, Cholesky is fast.
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Step size control parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 2 / mueff) / ((dim + 2)**2 + mueff)))
        
        # Damping
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initial best
        x_best = m.copy()
        fx_best = func(x_best)
        calls = 1
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sample population
            population = []
            for i in range(lam):
                # Sample z
                z = rng.standard_normal(dim)
                # Transform using C
                # For efficiency, we can update C and use Cholesky each step?
                # Or just use the current C.
                # Since dim is small, let's do Cholesky every step for simplicity/robustness.
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(dim)
                y = m + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                population.append(y)
            
            # Evaluate
            fitnesses = []
            for y in population:
                fy = func(y)
                fitnesses.append(fy)
                calls += 1
                if calls >= budget:
                    break
            
            if calls >= budget:
                # Update best with last few
                for i, fy in enumerate(fitnesses):
                    if fy < fx_best:
                        fx_best = fy
                        x_best = population[i].copy()
                break
            
            # Sort
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
            
            # Update best
            if fitnesses[0] < fx_best:
                fx_best = fitnesses[0]
                x_best = population[0].copy()
                
            # Update mean
            m_old = m.copy()
            for i in range(mu):
                m += weights[i] * population[i]
                
            # Update sigma
            # Evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / (sigma * np.linalg.norm(m - m_old) + 1e-10) * (m - m_old)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (budget - calls))) < (1.4 + 2 / (dim + 1)) * chi_n
            if hsig:
                # Update pc
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) / (sigma * np.linalg.norm(m - m_old) + 1e-10) * (m - m_old)
                
                # Update C
                v = (m - m_old) / sigma
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(v, v) + (1 - hsig) * c_c * (2 - c_c) * np.outer(pc, pc))
                for i in range(mu):
                    wi = weights[i]
                    vi = (population[i] - m_old) / sigma
                    C += c_mu * wi * np.outer(vi, vi)
                    
                # Ensure positive definiteness
                C = (C + C.T) / 2
                # Regularize if needed
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-10:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
                    
                # Update sigma
                sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-5, 10.0)
                
                # Update mean
                m = np.clip(m, -5, 5)
                
                # Restart if sigma is too small or too large?
                if sigma < 1e-4 or sigma > 100:
                    # Simple restart
                    m = rng.uniform(-5, 5, dim)
                    sigma = 2.0
                    C = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    # Don't reset best, but restart local search
