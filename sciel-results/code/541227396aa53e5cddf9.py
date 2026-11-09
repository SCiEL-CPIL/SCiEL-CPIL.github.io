"""Optimized CMA-ES for BBOB f11 Discus."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0  # 10 seconds safety margin before 330s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 10  # Population size
        mu = lam // 2  # Number of recombination weights
        w = np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        w = np.concatenate([w, np.zeros(lam - mu)])
        mu_eff = 1.0 / np.sum(w[:mu]**2)
        
        # Step size initialization
        x0 = np.zeros(dim)
        # Discus functions often have a global minimum shifted. 
        # Starting at 0 is a reasonable heuristic as the minimizer is in [-4,4]^5.
        # We could sample a few points to pick a better start, but that costs budget.
        # Let's use a small random start to avoid symmetry issues if minimizer is not at 0.
        # However, f11 is symmetric around the minimum. Let's try starting near 0.
        x = np.zeros(dim)
        
        # Initial evaluation
        fvals = [func(x)]
        best_val = fvals[0]
        best_x = x.copy()
        calls_made = 1
        
        # If we run out of budget immediately, return
        if calls_made >= budget:
            return best_x, best_val

        sigma = 1.0
        # CMA-ES parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / dim - 1)) + c_sigma
        c_c = (4 + dim / lam) / (dim + 4 + 2 * mu_eff / (lam - 2))
        c_1 = 2 / (dim**1.7 + 10)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/(dim+2)) / (dim**1.7 + lam + 10))
        
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Pre-compute inverse sqrt of C using eigendecomposition initially (identity)
        # In the loop, we update C and recompute C_inv_sqrt
        
        # Main loop
        while calls_made < budget and (time.time() - start_time) < time_limit:
            # Generate lambda search points
            ys = np.array([np.random.standard_normal(dim) for _ in range(lam)])
            zs = np.dot(C_inv_sqrt, ys.T).T
            
            # Update step size (sigma)
            # Calculate mean of selected weights
            # Sort by fitness
            # We need to evaluate all lambda points first
            
            candidates = []
            for i in range(lam):
                xi = x + sigma * zs[i]
                candidates.append(xi)
            
            # Evaluate candidates
            for i, xi in enumerate(candidates):
                if calls_made >= budget or (time.time() - start_time) > time_limit:
                    break
                fi = func(xi)
                calls_made += 1
                fvals.append(fi)
                if fi < best_val:
                    best_val = fi
                    best_x = xi.copy()
            
            # If we broke early due to budget/time, we might have fewer than lam new points
            # We need to ensure we have enough points to sort and select
            # If we don't have lam new points, we can't do standard CMA-ES update properly.
            # Let's check how many new points we have.
            new_fvals = fvals[-len(candidates):]
            
            # If we ran out of budget mid-generation, pad with infinity? 
            # Or just stop the adaptation and return best.
            if len(new_fvals) < lam:
                # Pad with inf to maintain structure if needed, but really we should stop
                # Since the loop condition checks budget, if we broke, we are done or close.
                # Let's just break the main loop if we didn't complete a generation.
                break
                
            # Sort by fitness
            indices = np.argsort(new_fvals)
            xs = np.array([candidates[i] for i in indices])
            fs = np.array(new_fvals[indices])
            
            # Update mean (recombination)
            x_old = x.copy()
            x = np.sum([w[i] * xs[i] for i in range(lam)], axis=0)
            
            # Update covariance adaptation
            # Delta rule for pc
            pc = (1 - c_c) * pc + c_c * (x - x_old) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x - x_old)
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made - lam)))) < (1.4 + 2/(dim+1)) * chi_n
            if hsig:
                ps = 0.0
            
            # Rank-one and rank-mu update
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                yi = (xs[i] - x_old) / sigma
                rank_mu += w[i] * np.outer(yi, yi)
            
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0) # Cap sigma to prevent explosion
            
            # Update C_inv_sqrt
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10) # Numerical stability
            C_inv_sqrt = np.dot(eigvecs, np.diag(1 / np.sqrt(eigvals)))
            # C_sqrt is not strictly needed for generation, only C_inv_sqrt is used for z = C_inv_sqrt @ y

        return best_x, best_val
# EVOLVE-BLOCK-END
