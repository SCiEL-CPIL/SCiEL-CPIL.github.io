import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        if lam < 2: lam = 2
        mu = lam // 2
        
        # Weights
        logw = np.log(np.arange(mu, 0, -1))
        weights = logw / np.sum(logw)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial values
        x = np.zeros(dim)
        sigma = 2.0 
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Track best
        best_x = x.copy()
        best_f = func(x)
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        count = 1
        gen = 0
        
        while count + lam <= budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = np.zeros((lam, dim))
            fs = np.zeros(lam)
            
            # Pre-compute Cholesky for this generation
            L = np.linalg.cholesky(C)
            
            for i in range(lam):
                z = np.random.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                count += 1
                
            # Update best
            min_idx = np.argmin(fs)
            if fs[min_idx] < best_f:
                best_f = fs[min_idx]
                best_x = ys[min_idx].copy()
                
            # Sort individuals
            order = np.argsort(fs)
            ys_sorted = ys[order]
            
            # Update mean
            new_x = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
            
            # Update sigma
            pc = (1 - c_sigma) * pc + c_sigma * (new_x - x) / sigma
            # gen is the current generation index (0-based)
            # The denominator in hsig uses the number of generations so far.
            # Standard formula: sqrt(1 - (1-c_sigma)^(2*t)) where t is generation count.
            # Here gen is 0 for first update. We use gen+1 or just gen?
            # Usually t starts at 1.
            t = gen + 1
            hsig = np.linalg.norm(pc) / np.sqrt(1 - (1 - c_sigma)**(2 * t)) / chi_n < 1.4 + 2 / (dim + 1)
            
            ps = (1 - c_sigma) * ps + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (new_x - x)
            
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            delta_sig = hsig * (1 + c_sigma) - 1
            delta_hsig = (1 - hsig) * (1 + c_sigma)
            
            C = (1 + c_1 * delta_sig + c_mu * delta_hsig) * C \
                - c_1 * np.outer(pc, pc) \
                - c_mu * np.sum(weights[:, None, None] * np.outer(ys_sorted[:mu] - x, ys_sorted[:mu] - x) / sigma**2, axis=0)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-8 * np.eye(dim)
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
            
            x = new_x
            gen += 1

        # Use remaining budget if any
        while count < budget:
            if time.time() > deadline:
                break
            # Simple random step from best
            y = best_x + 0.1 * np.random.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            count += 1
            if fy < best_f:
                best_f = fy
                best_x = y

        return best_x, best_f
