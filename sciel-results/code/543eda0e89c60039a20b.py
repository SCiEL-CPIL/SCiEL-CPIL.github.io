# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    CMA-ES adaptively models the search distribution's covariance to align 
    with the anisotropic curvature of the Bent Cigar function, allowing 
    efficient navigation of the narrow valley without gradient information.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def wrapper(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f_val = float(func(x_clipped))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return float('inf')

        # CMA-ES Parameters
        # Standard CMA-ES settings for low dimensions
        lambda_ = 4 + int(3 * np.log(dim))  # Population size
        mu = lambda_ // 2                     # Number of selected points
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / (w @ w)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial state
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covers the box [-5, 5] reasonably well
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        invsqrtC = np.linalg.inv(np.sqrtm(C))
        
        # Precompute Cholesky factor for sampling
        # We will update C and recompute Cholesky each generation for simplicity, 
        # as dim=5 is small.
        
        while calls < budget and time.time() < deadline:
            # Generate population
            pop = []
            for _ in range(lambda_):
                z = np.random.multivariate_normal(np.zeros(dim), C)
                x = mean + sigma * z
                x = np.clip(x, -5, 5)
                pop.append(x)
            
            # Evaluate
            fvals = np.array([wrapper(x) for x in pop])
            
            # Check if we ran out of budget mid-generation
            if calls >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(fvals)
            pop = np.array(pop)
            fvals = fvals[idx]
            pop = pop[idx]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()
            
            # Update mean
            new_mean = sum(w[i] * pop[i] for i in range(mu))
            
            # Update covariance matrix adaptation
            # Calculate difference vectors
            y_w = (new_mean - mean) / sigma
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * y_w
            cs_norm = np.linalg.norm(pc)
            
            # Update C
            delta_hs = (cs_norm / np.sqrt(1 - (1 - cs) ** (2 * (calls // lambda_ + 1)))) if calls // lambda_ > 0 else 0
            # Simplified delta_hs for implementation stability
            delta_hs = 1.0 if cs_norm / np.sqrt(1 - (1 - cs) ** (2 * (calls // lambda_ + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((1 + cs) / (1 - cs)) else 0.0
            
            C = (1 - c1 - cmu) * C + c1 * ((1 - delta_hs) * (2 / (1 - (1 - cs) ** (2 * (calls // lambda_ + 1))) - 1) * np.outer(pc, pc) + delta_hs * np.eye(dim)) \
                + cmu * sum(w[i] * np.outer(pop[i] - mean, pop[i] - mean) / sigma**2 for i in range(mu))
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small identity to ensure PD
            C += 1e-12 * np.eye(dim)
            
            # Update sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * invsqrtC @ y_w
            sigma *= np.exp(cs / (dim + 2) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls // lambda_ + 1))) - 1))
            
            # Update mean
            mean = new_mean
            
            # Recompute inverse square root of C for next iteration
            try:
                # Use Cholesky for stability
                L = np.linalg.cholesky(C)
                invsqrtC = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback to eigen if Cholesky fails
                vals, vecs = np.linalg.eigh(C)
                vals = np.clip(vals, 1e-12, None)
                invsqrtC = vecs @ np.diag(1 / np.sqrt(vals)) @ vecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
