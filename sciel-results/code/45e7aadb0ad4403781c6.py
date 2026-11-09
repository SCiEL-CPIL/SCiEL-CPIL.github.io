"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Lambda is population size
        # For n=5, lambda = 4 + floor(3*log(5)) = 4 + 4 = 8. 
        # Let's use 10 for slightly better exploration.
        lambda_pop = 10
        mu = lambda_pop // 2  # Selection size
        
        # Weights
        # Standard weights: w_i = log(mu + 0.5) - log(i) for i=1..mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Damping
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / n - 1) - 1) + c_sigma
        
        # Initial state
        # Start in the center of the box
        m = np.zeros(n)
        # Initial sigma: scale to cover the box [-5, 5]
        # A sigma of 1.0 covers +/- 3 sigma = 3.0, which is a bit small.
        # Let's start with sigma = 2.0 to cover a larger area initially.
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Precompute Cholesky factor of C
        # We need to update C and its Cholesky factor
        # For simplicity and robustness, we can recompute Cholesky every step or use a stable update.
        # Given n=5, recomputing is very cheap.
        
        calls_made = 0
        best_x = m.copy()
        best_f = float('inf')
        
        # We need to ensure we don't exceed budget
        # Each iteration uses lambda_pop calls
        
        while calls_made + lambda_pop <= budget or (calls_made < budget and calls_made + lambda_pop > budget):
             # If remaining budget is less than lambda_pop, we can either:
             # 1. Stop (if we want strict batches)
             # 2. Do a partial batch (greedy)
             # The contract says "More than 500 calls ... makes the whole result invalid".
             # It does NOT say we must use batches of size lambda_pop.
             # However, CMA-ES relies on the batch. 
             # If we have 500 budget and lambda=10, we do exactly 50 iterations.
             # If budget was 505, we could do 50 full iterations and 5 greedy steps?
             # Standard CMA-ES implementations often just stop if not enough for a full batch.
             # Let's check: 500 / 10 = 50. So it fits perfectly.
             
             if calls_made >= budget:
                 break
             
             # Generate population
             # We need the square root of C
             # Use Cholesky decomposition
             try:
                 L = np.linalg.cholesky(C)
             except np.linalg.LinAlgError:
                 # If C is not positive definite, regularize
                 C = C + 1e-10 * np.eye(n)
                 L = np.linalg.cholesky(C)
                 
             # Generate standard normal samples
             z = rng.standard_normal((lambda_pop, n))
             
             # Transform to candidates
             # x_i = m + sigma * z_i @ L.T (since z is row vector, we need L.T to apply to columns? 
             # Cov = sigma^2 * L @ L.T. 
             # If z is (lambda, n), we want x = m + sigma * z @ L.T ? 
             # Var(z @ L.T) = L * I * L.T = L @ L.T = C. Correct.
             pop = m + sigma * (z @ L.T)
             
             # Clip to box? The function clips, but we can clip here to be safe/consistent.
             # The problem says func clips. It's safer to pass raw values if func handles it,
             # but clipping here ensures we don't waste time on points that are effectively the same.
             # Actually, if I clip, the distribution of evaluated points is distorted.
             # Best practice: Let func clip.
             
             # Evaluate
             fvals = np.array([func(x) for x in pop])
             calls_made += lambda_pop
             
             # Sort indices by fitness
             order = np.argsort(fvals)
             pop = pop[order]
             fvals = fvals[order]
             
             # Update best
             if fvals[0] < best_f:
                 best_f = fvals[0]
                 best_x = pop[0].copy()
             
             # Update Mean
             # m_new = m + sigma * sum(w_i * (x_i - m) / sigma) ? No.
             # m = m + sigma * sum(w_i * z_i) is not quite right if we use x.
             # Standard formula: m = m + sigma * sum(w_i * (x_i - m) / sigma) ?
             # Actually, x_i = m + sigma * y_i where y_i are correlated normals.
             # So x_i - m = sigma * y_i.
             # m_new = m + sum(w_i * (x_i - m))
             
             m = m + np.sum(weights[:, np.newaxis] * (pop[:mu] - m), axis=0)
             
             # Update Evolution Paths
             # pc = (1 - c_c) * pc + c_c * sqrt(mu_eff) * sigma * sum(w_i * y_i)
             # y_i = (x_i - m_old) / sigma
             # We need m_old. Let's store it.
             
             # Wait, I updated m already. I need m_old for the path update.
             # Let's re-structure.
             
             # This implementation is getting complex to do in-place without careful bookkeeping.
             # Let's use a simpler, robust CMA-ES variant or a library-like implementation.
             # Since I cannot import cma, I will write it carefully.
             
             # Actually, let's restart the loop logic with proper state saving.
             
             # I will rewrite the loop content to be cleaner.
             
             pass # Placeholder for logic correction in the final edit

# The above thought process reveals that writing a full CMA-ES from scratch in a single pass without errors is tricky.
# I will provide a complete, correct implementation in the SEARCH/REPLACE block.
# I will use the standard update formulas.
# 1. Save m_old
# 2. Generate pop
# 3. Evaluate
# 4. Sort
# 5. Update m
# 6. Update pc, ps, sigma, C
# 7. Check budget

# Let's refine the code structure.

# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_pop = 10
        mu = lambda_pop // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Damping
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / n - 1) - 1) + c_sigma
        
        # Initial state
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        calls_made = 0
        best_x = m.copy()
        best_f = float('inf')
        
        # Main Loop
        while calls_made < budget:
            # Check if we can do a full batch
            # If not, we break to avoid exceeding budget.
            # 500 % 10 == 0, so this is safe.
            if budget - calls_made < lambda_pop:
                break
                
            # Save old mean for path updates
            m_old = m.copy()
            
            # Generate population
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-10 * np.eye(n)
                L = np.linalg.cholesky(C)
                
            z = rng.standard_normal((lambda_pop, n))
            pop = m + sigma * (z @ L.T)
            
            # Evaluate
            fvals = np.array([func(x) for x in pop])
            calls_made += lambda_pop
            
            # Sort
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()
                
            # Update Mean
            m = m + np.sum(weights[:, np.newaxis] * (pop[:mu] - m_old), axis=0)
            
            # Update Evolution Paths
            # y_i = (x_i - m_old) / sigma
            y = (pop[:mu] - m_old) / sigma
            # pc = (1 - c_c) * pc + c_c * sqrt(mu_eff) * sum(w_i * y_i)
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * np.sum(weights[:, np.newaxis] * y, axis=0)
            
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m - m_old) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m - m_old) / sigma
            
            # Update Sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||ps|| / E||N(0, I)|| - 1))
            # E||N(0, I)|| = sqrt(n) * (1 - 1/(4n) + 1/(21n^2)) approx
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + lambda_pop) / (n * d_sigma)))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
            # Note: The exponent for (1-c_sigma) is based on generations.
            # calls_made is total evaluations. Generations = calls_made / lambda_pop.
            # Let's use generations count.
            gen = calls_made // lambda_pop
            # The formula for hsig is slightly different in some implementations.
            # Standard: hsig = ||ps|| / (E||N(0,I)|| * sqrt(1 - (1-c_sigma)^{2*gen})) < (1.4 + 2/(n+1)) * sqrt((n+1)/(n-1))
            
            # Simpler hsig check often used:
            # hsig = np.linalg.norm(ps) < (1.4 + 2.0 / (n + 1.0)) * np.sqrt((n + 1.0) / (n - 1.0))
            # This ignores the cumulative effect, which is an approximation but often sufficient.
            # However, the cumulative effect is important for stability.
            # Let's use the standard approximation with the current generation count.
            
            norm_ps = np.linalg.norm(ps)
            c_sigma_factor = np.sqrt(1.0 - (1.0 - c_sigma)**(2 * gen))
            if c_sigma_factor > 1e-10:
                hsig = norm_ps / c_sigma_factor < (1.4 + 2.0 / (n + 1.0)) * np.sqrt((n + 1.0) / (n - 1.0))
            else:
                hsig = True
                
            e_norm = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))
            
            sigma = sigma * np.exp((c_sigma / d_sigma) * (norm_ps / e_norm - 1.0))
            
            # Update Covariance Matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - hsig) * c_c * (2 - c_c) * I) + c_mu * sum(w_i * y_i * y_i.T)
            
            delta_hsig = (1.0 - hsig) * c_c * (2.0 - c_c)
            C = (1.0 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_hsig * np.eye(n)) + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('i,ij,ik->ijk', y, y, y), axis=0)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2.0
            # Regularize if needed
            # Check eigenvalues
            # For n=5, we can just add small identity if Cholesky fails later
            
        return best_x, best_f
# EVOLVE-BLOCK-END
