import numpy as np
import time

class Optimizer:
    """CMA-ES for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 285

        best_x = None
        best_f = float('inf')
        calls = 0

        # CMA-ES Parameters
        n = self.dim
        # Population size
        lam = 4 + int(3 * np.log(n))
        if lam < 10: lam = 10 # Ensure sufficient diversity for n=5
        
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Step size adaptation
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial step size
        sigma = 2.0 # Box is [-5, 5], so 2.0 is a reasonable start
        
        # Initial mean
        m = np.zeros(n)
        
        # State variables
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Restart logic
        max_restarts = 5
        calls_per_restart = self.budget // max_restarts
        
        for restart in range(max_restarts):
            if calls >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Reset state for restart
            # For f24, restarting near the center or best found is good
            if best_x is not None:
                m = best_x.copy()
            else:
                m = rng.uniform(-2, 2, n)
            sigma = 1.5
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Run CMA-ES for a fixed number of generations or until budget
            gen_calls = 0
            while calls < self.budget and gen_calls < calls_per_restart:
                if time.time() > deadline:
                    break
                
                # Generate population
                Y = []
                F = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    # Whitening transformation: C^{1/2} z
                    # For n=5, we can use eigen decomposition or just use Cholesky if positive definite
                    # Simpler: use the standard CMA-ES update which maintains C
                    # y = m + sigma * (C^{1/2} z)
                    # We need to compute C^{1/2}. For small n, this is fast.
                    # Actually, standard CMA-ES uses the matrix C directly in updates,
                    # but sampling requires the square root.
                    # A simpler approach for small n: use the eigen decomposition.
                    # Or, since we are in Python, we can just use the standard update rules.
                    # Let's use a simplified version: sample from N(m, sigma^2 C).
                    # np.random.multivariate_normal(m, sigma**2 * C, size=lam)
                    # This is easy and correct.
                    
                    pass # Placeholder for sampling
                
                # Let's use np.random.multivariate_normal for simplicity and correctness
                # It might be slightly slower but n=5 is small.
                
                samples = rng.multivariate_normal(m, sigma**2 * C, size=lam)
                # Clip to box
                samples = np.clip(samples, -5, 5)
                
                # Evaluate
                for i in range(lam):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    f_val = func(samples[i])
                    calls += 1
                    Y.append(samples[i])
                    F.append(f_val)
                    
                    if f_val < best_f:
                        best_f = f_val
                        best_x = samples[i].copy()
                
                if len(F) < lam:
                    break
                    
                # Sort by fitness
                idx = np.argsort(F)
                Y = np.array(Y)
                F = np.array(F)
                Y = Y[idx]
                F = F[idx]
                
                # Update mean
                m_old = m.copy()
                m = np.sum(weights[:, None] * Y[:mu], axis=0)
                
                # Update step size
                # h_sigma = (norm(ps) / sqrt(1 - (1 - 2*gen/lam)**(2*c_sigma)) ) / chi_n < 1.4 + 2/(n+1)
                # Simplified h_sigma check
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ (m - m_old))
                h_sigma = (np.linalg.norm(ps) / chi_n) < (1.4 + 2 / (n + 1))
                
                # Update pc
                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - m_old) / sigma
                
                # Update C
                delta_h = (h_sigma - 1) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_h - c_1) * C + c_1 * np.outer(pc, pc) + c_mu * np.sum(weights[:mu][:, None, None] * np.outer(Y[:mu] - m_old, Y[:mu] - m_old) / sigma**2, axis=0)
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Check for stagnation
                if np.linalg.norm(m - m_old) < 1e-10 and sigma < 1e-6:
                    break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
