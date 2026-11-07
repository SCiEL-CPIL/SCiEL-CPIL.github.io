"""CMA-ES for BBOB f23 (Katsuura) optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # Global best tracking
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls = 1
        
        # CMA-ES parameters
        # For dim=5, lambda = 4 + floor(3 * log(5)) = 4 + 4 = 8
        lam = 4 + int(3 * np.log(self.dim))
        
        # Initial sigma
        sigma = 2.0  # Larger initial step to explore the [-5,5] box quickly
        
        # Mean vector
        m = x_best.copy()
        
        # Covariance matrix C
        C = np.eye(self.dim)
        
        # Eigen decomposition cache
        D = np.ones(self.dim)  # Diagonal matrix of sqrt(eigenvalues)
        B = np.eye(self.dim)   # Rotation matrix
        
        # Step size control parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # Cumulation step size
        sigma0 = 2.0
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        # History for cumulation
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        # Track best
        x_best = m.copy()
        f_best = func(x_best)
        calls += 1
        
        # Main loop
        while calls < self.budget and time.time() < deadline:
            # Generate population
            pop = np.zeros((lam, self.dim))
            for i in range(lam):
                z = rng.standard_normal(self.dim)
                pop[i] = m + sigma * (B @ (D * z))
            
            # Evaluate population
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls >= self.budget or time.time() > deadline:
                    break
                # Clip to box
                x_eval = np.clip(pop[i], -5, 5)
                fvals[i] = func(x_eval)
                calls += 1
                # Update global best
                if fvals[i] < f_best:
                    f_best = fvals[i]
                    x_best = x_eval.copy()
            
            # If we ran out of budget or time mid-population, break
            if calls >= self.budget or time.time() > deadline:
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            y = pop[order]
            f_sorted = fvals[order]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, None] * y[:mu], axis=0)
            
            # Update cumulation for sigma
            # h_sigma condition
            # We need B and D for the transformation
            # Recalculate if not done recently? For simplicity in this implementation,
            # we update C every step and diagonalize every step (dim=5 is small)
            # Or we can approximate. Let's do standard update.
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (sigma * chi_n) * (B.T @ (D @ (B.T @ (m - m_old))))
            
            # Update p_c
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - lam))) < (1.4 + 2 / (self.dim + 1)) * chi_n)
            h_sigma = 1 if h_sigma else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - m_old) / sigma
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_c) * C + c_1 * (p_c @ p_c - C) + c_mu / mu_eff * np.sum(weights[:, None, None] * (y[:mu] - m_old)[:, None, :] * (y[:mu] - m_old)[None, :, :], axis=0)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Ensure sigma doesn't explode or vanish
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Diagonalize C to get B and D
            # Use eigh for symmetric matrix
            eigvals, eigvecs = np.linalg.eigh(C)
            # Filter out negative eigenvalues due to numerical errors
            eigvals = np.maximum(eigvals, 1e-12)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Check for convergence or stall
            # If sigma is very small or no improvement in a while, we could restart,
            # but CMA-ES is generally robust. Let's just continue.
            
        return x_best, f_best
# EVOLVE-BLOCK-END
