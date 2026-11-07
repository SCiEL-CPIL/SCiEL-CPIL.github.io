"""CMA-ES optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation
        cs = (mueff + 1) / (dim + mueff + 2)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        
        # Covariance matrix adaptation
        c1 = (2 / ((dim + 1.3)**2 + mueff)) * (1 - cs) / (1 + cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initialization
        rng = np.random.default_rng(42)
        
        # Start near the center, but slightly random to avoid symmetry issues if any
        # The problem says minimizer is in [-4, 4]^5, so 0 is a good start
        xmean = np.zeros(dim)
        
        # Initial sigma
        sigma = 1.0
        
        # Initial covariance
        C = np.eye(dim)
        Cinv = np.eye(dim)
        Csqrt = np.eye(dim)
        Cinv_sqrt = np.eye(dim)
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # History for progress monitoring
        best_x = xmean.copy()
        best_f = np.inf
        
        # Initial evaluation
        # We need to evaluate the initial point
        # Note: func clips to [-5, 5]
        fvals = []
        xs = []
        
        # We will maintain a list of evaluations to track best
        # But CMA-ES works in batches.
        
        # To ensure we use the budget efficiently and handle the "first call" exception rule,
        # we make the first call explicitly.
        
        # Helper to evaluate and track
        def evaluate(x):
            nonlocal best_x, best_f
            fx = func(np.clip(x, -5, 5))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial point
        x0 = xmean.copy()
        f0 = evaluate(x0)
        budget_used = 1
        
        # Main loop
        while budget_used < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            # Update Cinv_sqrt if needed (we do it every generation for simplicity, 
            # though every 10 generations is standard. Given small dim, every gen is fine)
            # Actually, we need Cinv_sqrt for sampling.
            # Let's compute Cinv_sqrt at the start of the loop.
            # But C changes every gen.
            
            # Standard CMA-ES loop:
            # 1. Sample
            # 2. Evaluate
            # 3. Update
            
            # We need to be careful with the budget.
            # Each iteration consumes `lam` evaluations.
            # If remaining budget < lam, we can do a partial run or just stop.
            # For simplicity, if remaining < lam, we break.
            if budget - budget_used < lam:
                break
                
            # Compute Cinv_sqrt
            # Using eigh for stability
            evals, evecs = np.linalg.eigh(C)
            # Clip evals to avoid negative due to numerical errors
            evals = np.maximum(evals, 1e-12)
            Cinv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
            
            # Sample
            ys = []
            for i in range(lam):
                # Sample from N(0, C)
                z = rng.standard_normal(dim)
                y = xmean + sigma * (Cinv_sqrt @ z)
                ys.append(y)
            
            # Evaluate
            fvals = np.array([evaluate(y) for y in ys])
            budget_used += lam
            
            # Sort
            idx = np.argsort(fvals)
            ys = np.array(ys)[idx]
            fvals = fvals[idx]
            
            # Update xmean
            xmean_new = xmean + sigma * sum(weights[i] * ys[i] for i in range(mu))
            
            # Update p_sigma
            # Need Cinv_sqrt for previous C? No, p_sigma uses the normalized direction.
            # The direction is (xmean_new - xmean) / sigma
            # But we need to project it? No, standard CMA-ES uses:
            # p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mueff) * (Cinv_sqrt_old @ (xmean_new - xmean) / sigma)
            # Wait, the standard formula uses the inverse square root of the OLD C.
            # Since we just computed Cinv_sqrt for the NEW C (which is updated after), 
            # we should have stored the old one or computed it before updating C.
            # Let's re-order:
            # 1. Compute Cinv_sqrt from current C
            # 2. Sample
            # 3. Evaluate
            # 4. Update xmean
            # 5. Update p_sigma using current Cinv_sqrt (which corresponds to old C)
            # 6. Update sigma
            # 7. Update C
            # 8. Update xmean = xmean_new
            
            # We already have Cinv_sqrt from the start of the loop (based on old C).
            d = (xmean_new - xmean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_sqrt @ d)
            
            # Update sigma
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
            
            # Update C
            # Rank 1 update
            y = (xmean_new - xmean) / sigma
            rank1 = np.outer(y, y)
            
            # Rank mu update
            # y_i are the selected points, scaled
            # We need the old xmean
            y_selected = (ys[:mu] - xmean) / sigma
            # Weighted sum of outer products
            rankmu = sum(weights[i] * np.outer(y_selected[i], y_selected[i]) for i in range(mu))
            
            C = (1 + c1 + cmu - c1) * C + c1 * (rank1 - (1 - (1 - cs)**2) * C) + cmu * (rankmu - C)
            # Simplified: C = (1 + c1 + cmu) * C - c1 * (1 - (1-cs)**2) * C ... 
            # Standard formula:
            # C = (1 + c1 + cmu - c1) * C + c1 * (y y^T + (1 - (1-cs)^2) C) + cmu * sum(w_i y_i y_i^T)
            # Wait, the standard update is:
            # C <- (1 + c1 + cmu) * C - c1 * (1 - (1-cs)^2) * C ... no.
            # C <- (1 - c1) * C + c1 * (y y^T + (1 - (1-cs)^2) C) ... no.
            
            # Correct formula from Hansen's thesis:
            # C = (1 + c1 + cmu) * C - c1 * (1 - (1-cs)^2) * C ... 
            # Actually:
            # C = (1 - c1) * C + c1 * (y y^T + (1 - (1-cs)^2) C) + cmu * H
            # where H = sum(w_i y_i y_i^T)
            # This simplifies to:
            # C = (1 - c1 + c1*(1-(1-cs)^2)) * C + c1 * y y^T + cmu * H
            # Coefficient of C: 1 - c1 + c1 - c1(1-2cs+cs^2) = 1 - c1(1-2cs+cs^2) = 1 - c1(1-cs)^2
            
            C = (1 - c1 * (1 - cs)**2) * C + c1 * np.outer(y, y) + cmu * rankmu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update xmean
            xmean = xmean_new
            
            # Optional: Check for stagnation
            # If sigma is very small, we might be stuck.
            # But CMA-ES is quite robust.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
