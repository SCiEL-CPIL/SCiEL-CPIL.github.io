import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 250.0  # Safety margin

        # Phase 1: Global Search (Random Sampling)
        # Use 10% of budget for global exploration
        n_global = max(10, budget // 10)
        remaining_budget = budget - n_global
        
        best_x = None
        best_f = float('inf')
        
        # Random initialization
        # We can try a few random starts to be safe, but 10 random points is usually enough to get near the valley in 5D
        for i in range(n_global):
            if time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no calls were made (shouldn't happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            remaining_budget = budget - 1

        # Phase 2: CMA-ES Local Search
        # Initialize CMA-ES parameters
        # Lambda (population size)
        lam = 4 + int(3 * np.log(dim))
        if lam < 4: lam = 4
        if lam > budget: lam = budget
        
        # Mu (number of parents)
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size
        sigma = 2.0
        
        # Mean
        mean = best_x.copy()
        
        # Covariance
        C = np.eye(dim)
        
        # Inverse square root of C (for sampling)
        # We use Cholesky decomposition
        L = np.linalg.cholesky(C)
        
        # Evolution path for sigma
        ps = np.zeros(dim)
        # Evolution path for covariance
        pc = np.zeros(dim)
        
        # Parameters
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        c1 = (2 / ((dim + 1.3)**2 + mu_eff)) * (1 - c_sigma() if False else 1) # Simplified c1
        # Standard CMA-ES c1 formula: (2 / ((dim + 1.3)^2 + mu_eff)) * (1 - c_sigma) is not quite right.
        # c1 = (2 / ((dim + 1.3)**2 + mu_eff)) * (1 - c_sigma) is for some variants.
        # Standard: c1 = (2 / ((dim + 1.3)**2 + mu_eff))
        # cmu = (2 * (mu_eff - 2 + 1/mu_eff)) / ((dim + 2)**2 + mu_eff)
        
        c1 = (2.0 / ((dim + 1.3)**2 + mu_eff))
        cmu = (2.0 * (mu_eff - 2.0 + 1.0/mu_eff)) / ((dim + 2.0)**2 + mu_eff)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Correction factor for sampling
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        calls_made = n_global
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Determine how many to sample this iteration
            # We can sample up to lam, but must respect remaining budget
            n_sample = min(lam, budget - calls_made)
            if n_sample <= 0:
                break
            
            # Sample points
            offspring = np.zeros((n_sample, dim))
            for i in range(n_sample):
                z = np.random.standard_normal(dim)
                # Transform z using Cholesky factor L
                # x = mean + sigma * L * z
                offspring[i] = mean + sigma * (L @ z)
                # Clip to box
                offspring[i] = np.clip(offspring[i], -5, 5)
                
            # Evaluate
            offspring_f = np.array([func(x) for x in offspring])
            calls_made += n_sample
            
            # Sort
            idx = np.argsort(offspring_f)
            offspring = offspring[idx]
            offspring_f = offspring_f[idx]
            
            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()
            
            # Update Mean
            # Only use top mu, but if n_sample < mu, use all
            n_parents = min(mu, n_sample)
            w_parents = w[:n_parents]
            w_parents = w_parents / np.sum(w_parents) # Renormalize if truncated
            
            new_mean = np.sum(offspring[:n_parents] * w_parents[:, np.newaxis], axis=0)
            
            # Update Covariance
            # pc = (1-c1)*pc + sqrt(c1*(2-c1)*mueff) * (new_mean - mean)/sigma
            pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mu_eff) * (new_mean - mean) / sigma
            
            # Update ps
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * L^-1 * (new_mean - mean)/sigma
            # L^-1 is inv(L)
            inv_L = np.linalg.inv(L)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_L @ (new_mean - mean) / sigma)
            
            # Update Sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update Covariance Matrix C
            # h_sigma = (np.linalg.norm(ps) / np.sqrt(1-(1-c_sigma)^(2*(calls_made/lam+1)))) < 1.4 + 2/(dim+1)
            # This is complex to track exactly without generation count. 
            # Approximation: check if ps is small enough.
            h_sigma = 1.0 # Simplified, assuming it's usually 1 in early stages or we skip the correction for simplicity in this constrained budget.
            # Better: 
            # h_sigma = 1 if np.linalg.norm(ps) < (1.4 + 2/(dim+1)) * chi_n else 0
            h_sigma = 1.0 if np.linalg.norm(ps) < (1.4 + 2.0/dim) * chi_n else 0.0
            
            # C = (1 - c1 - cmu)*C + c1*(pc*pc.T + (1-h_sigma)*c1*(2-c1)*C) + cmu * sum(wi * (yi-mean)(yi-mean).T)
            # Note: The term (1-h_sigma)*c1*(2-c1)*C is for the "delta" correction.
            
            delta = np.zeros((dim, dim))
            if h_sigma < 1.0:
                delta = c1 * (2 - c1) * C
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta)
            
            for i in range(n_parents):
                y = offspring[i] - mean
                C += cmu * w_parents[i] * np.outer(y, y)
                
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            # Re-cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small identity if not PD
                C += 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)
            
            mean = new_mean

        return best_x, best_f
