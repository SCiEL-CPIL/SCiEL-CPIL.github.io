"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin well before 330s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Standard lambda for CMA-ES: lambda = 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + floor(3*1.609) = 4 + 4 = 8
        # However, for small n and limited budget, smaller lambda is often better for exploration.
        # Let's use lambda = 4 + floor(3*log(n)) but cap it or adjust.
        # Actually, a common choice is lambda = 4 + floor(3*log(n)).
        # For n=5, lambda=8.
        # Let's try a slightly smaller population for faster convergence in low dims?
        # No, CMA-ES is robust. Let's stick to standard or slightly reduced.
        # Let's use lambda = 4 + int(3 * np.log(n))
        lam = 4 + int(3 * np.log(n))
        
        # Initial mean
        m = np.zeros(n)
        
        # Initial step size sigma
        # Standard initialization: sigma0 = 1.0 or based on initial spread.
        # Since the box is [-5,5] and min is in [-4,4], a sigma of 1.0 is reasonable.
        sigma = 1.0
        
        # Covariance matrix C
        C = np.eye(n)
        
        # Inverse covariance matrix
        C_inv = np.eye(n)
        
        # Precompute square root of C and C_inv for sampling and transformation
        # We need to update these as C changes.
        # For efficiency, we can use eigen decomposition or Cholesky.
        # Since n=5, even full SVD or Eigen is cheap.
        
        # Update strategies
        # Weights
        # Standard weights: w_i = log((mu + 0.5)/i) - average(log((mu + 0.5)/i) for i=1..mu)
        # mu = floor(lam / 2)
        mu = lam // 2
        w = np.zeros(lam)
        for i in range(mu):
            w[i] = np.log((mu + 0.5) / (i + 1))
        w_sum = np.sum(w)
        w = w / w_sum
        
        # Effective population size
        mueff = 1.0 / np.sum(w**2)
        
        # Update rates
        # cmu (covariance matrix adaptation rate)
        cmu = (2 + 2 * np.sqrt((mueff - 1) / (n + 1)) - 1) / ((n + 1)**2 + 2)
        # cc (cumulation rate for step size control)
        cs = (mueff + 2) / (n + mueff + 5)
        # c1 (weight for rank-one update)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        # cmu (weight for rank-mu update)
        cmu = (2 - c1 - cmu) / ((n + 2) + mueff)
        
        # Step size control
        # chi_n = sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Cumulation path
        ps = np.zeros(n)
        
        # Track best
        best_x = m.copy()
        best_f = np.inf
        calls_made = 0
        
        # Helper to update Cholesky factors or Eigen decomposition
        # For simplicity and robustness with small n, we'll recompute eigen decomposition
        # whenever C changes significantly or every few iterations.
        # Actually, C changes every iteration. 
        # We can maintain C and its inverse.
        # Sampling: x = m + sigma * z * L @ v, where L is Cholesky of C, v is standard normal.
        # But C is updated via rank updates.
        # Let's use a simpler approach for small n: 
        # Maintain C. To sample, compute Cholesky of C.
        # To update C, use the standard CMA-ES update rules.
        
        # Let's restructure to be more efficient.
        # We will maintain C and C_inv.
        # Sampling:
        # 1. Generate standard normal z.
        # 2. Transform z using Cholesky factor of C.
        
        # Initialize Cholesky factor
        L = np.linalg.cholesky(C)
        
        def sample_cma(m, sigma, L):
            z = np.random.standard_normal(n)
            # Transform z by L
            # x = m + sigma * (L @ z)
            return m + sigma * (L @ z)
        
        # Main loop
        while calls_made < budget:
            # Time check
            if time.time() > deadline:
                break
            
            # Generate population
            # We need to evaluate lam points.
            # Check if we have enough budget for a full generation.
            if calls_made + lam > budget:
                # If not enough for full generation, do a (1+1) step or break?
                # Let's just do a single step (1+1) style mutation using the current distribution.
                # Or break if we can't even do 1.
                if calls_made >= budget:
                    break
                
                # Generate 1 point
                z = np.random.standard_normal(n)
                x = m + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                f = func(x)
                calls_made += 1
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                
                # Update mean towards better point?
                # Simple (1+1) update:
                if f < func(m): # We don't have current f(m) stored easily, let's assume m is best or close.
                    # Actually, we should track f(m).
                    pass
                # For simplicity, if we can't do full CMA-ES, just do random search around m.
                # But let's try to finish the current generation if possible.
                # If we can't do a full generation, let's just do a few single steps.
                while calls_made < budget and time.time() < deadline:
                    z = np.random.standard_normal(n)
                    x = m + sigma * (L @ z)
                    x = np.clip(x, -5, 5)
                    f = func(x)
                    calls_made += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                break
            
            # Generate lam points
            pop = []
            fvals = []
            for i in range(lam):
                z = np.random.standard_normal(n)
                x = m + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                f = func(x)
                calls_made += 1
                pop.append(x)
                fvals.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            pop = np.array(pop)
            fvals = np.array(fvals)
            
            # Sort population by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update mean
            # m_new = sum(w_i * x_i) for i=1..mu
            m_old = m.copy()
            m = np.sum(w[:mu, None] * pop[:mu], axis=0)
            
            # Update step size (sigma)
            # Compute ps
            # ps = (1 - cs) * ps + cs * sqrt(mueff) * C_inv^0.5 * (m - m_old) / sigma
            # We need C_inv^0.5. Let's compute it from C_inv.
            # Or use the Cholesky of C_inv.
            # Let's compute C_inv^0.5 via eigen decomposition of C_inv.
            # Or simpler: use the fact that C_inv is symmetric.
            # For n=5, let's just compute eigen decomposition of C.
            # eigvals, eigvecs = np.linalg.eigh(C)
            # C_inv_sqrt = eigvecs @ np.diag(1/np.sqrt(eigvals)) @ eigvecs.T
            
            # To avoid recomputing eigen every time, we can update C and C_inv incrementally.
            # But let's start with a simple implementation that recomputes necessary factors.
            
            # Update C and C_inv
            # Rank-one update
            # C = (1 + cmu * (c1 + cmu - 1)) * C + cmu * (ps @ ps.T - C) + c1 * ((m - m_old) @ (m - m_old).T)
            # This formula is for the standard CMA-ES.
            
            # Let's compute C_inv_sqrt properly.
            # Since n is small, we can afford to do this.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to avoid division by zero or negative due to numerical errors
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            
            # Update ps
            ps = (1 - cs) * ps + cs * np.sqrt(mueff) * (C_inv_sqrt @ (m - m_old)) / sigma
            
            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update C
            # C = (1 + cmu * (c1 + cmu - 1)) * C + cmu * (ps @ ps.T - C) + c1 * ((m - m_old) @ (m - m_old).T)
            # Wait, the standard update is:
            # C = (1 + cmu * (c1 + cmu - 1)) * C + cmu * (ps @ ps.T - C) + c1 * (delta @ delta.T)
            # where delta = (m - m_old) / sigma
            
            delta = (m - m_old) / sigma
            C = (1 + cmu * (c1 + cmu - 1)) * C + cmu * (ps @ ps.T - C) + c1 * np.outer(delta, delta)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update C_inv
            # We can recompute C_inv from C using inverse.
            # Or update it incrementally.
            # For simplicity and robustness, let's recompute C_inv.
            C_inv = np.linalg.inv(C)
            
            # Update Cholesky factor L for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, add small regularization
                C += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(C)
                C_inv = np.linalg.inv(C)

        return best_x, best_f

# EVOLVE-BLOCK-END
