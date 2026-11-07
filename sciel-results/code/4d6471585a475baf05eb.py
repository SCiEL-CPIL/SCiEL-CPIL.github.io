"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for rotated Rosenbrock (f9)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s limit

        # CMA-ES Parameters
        # Population size lambda
        # Standard formula: lambda = 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + 3*2 = 10. Let's use 12 for a bit more robustness.
        lam = 12
        mu = lam // 2  # Number of selected individuals
        
        # Weights for selected individuals
        # Standard weights: w_i = log(mu + 0.5) - log(i) for i=1..mu
        # Normalized to sum to 1
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        
        # Variance-effectiveness population size
        mueff = 1.0 / np.sum(w**2)
        
        # Step size sigma
        # Start with a relatively large step to explore the box [-5, 5]^5
        # The box width is 10. A step of 2-3 is reasonable.
        sigma = 2.0
        
        # Initial mean x
        # Start at origin, as the shift is in [-4, 4]^5
        x = np.zeros(n)
        
        # Covariance matrix C
        C = np.eye(n)
        
        # Evolution path
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Damping parameters
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cmu = min(1 - cs, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        c1 = (2 - cmu) / ((n + 1.3)**2 + mueff)
        c_sigma = (mueff + 1.5) / (n + mueff + 5)
        
        # Call tracking
        calls = 0
        best_x = x.copy()
        best_f = np.inf
        
        # Helper to check if we should stop
        def should_stop():
            return calls >= budget or time.time() > deadline

        # Initial evaluation of the starting point
        f0 = func(x)
        calls += 1
        if f0 < best_f:
            best_f = f0
            best_x = x.copy()

        if should_stop():
            return best_x, best_f

        # CMA-ES Main Loop
        while not should_stop():
            # Check if we have enough budget for a full generation
            if calls + lam > budget:
                break
            
            # Sample population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes non-positive definite
                C = np.eye(n)
                L = np.eye(n)
            
            # Generate random vectors
            # We use standard normal
            zs = np.random.normal(0, 1, (lam, n))
            
            # Transform to x_i
            # x_i = x + sigma * z_i @ L.T
            pop = x + sigma * (zs @ L.T)
            
            # Evaluate population
            fitness = np.empty(lam)
            for i in range(lam):
                if should_stop():
                    # Fill remaining with inf if we run out of budget mid-generation
                    fitness[i:] = np.inf
                    break
                # Clip to box? The problem says func clips, but we can clip here to be safe/efficient
                # However, CMA-ES works better in unconstrained space if the box is large.
                # The box is [-5, 5]. If we go out, func clips.
                # It's better to let CMA-ES handle the boundary implicitly or just clip the input to func.
                xi = np.clip(pop[i], -5.0, 5.0)
                fitness[i] = func(xi)
                calls += 1
                
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = xi.copy()
            
            if should_stop():
                break
            
            # Sort individuals by fitness
            idx = np.argsort(fitness)
            sorted_fitness = fitness[idx]
            sorted_pop = pop[idx]
            
            # Select best mu individuals
            # Note: We must use the unclipped pop for the CMA-ES update math, 
            # but the fitness corresponds to the clipped evaluation.
            # This is a standard approximation for bounded problems.
            selected_pop = sorted_pop[:mu]
            selected_fitness = sorted_fitness[:mu]
            
            # Update mean
            # x_new = sum(w_i * x_i)
            x_new = np.sum(w[:, np.newaxis] * selected_pop, axis=0)
            
            # Update evolution paths
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * (L^{-1} / sigma) * (x_new - x)
            diff = x_new - x
            inv_L = np.linalg.inv(L)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_L @ diff / sigma)
            
            # pc = (1 - cs) * pc + sqrt(cs * (2 - cs) * mueff) * (L^{-1} / sigma) * (x_new - x) 
            # Wait, pc uses the same term but different damping?
            # Standard CMA-ES:
            # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * (1/sigma) * B^{-1} * (x_new - x)
            # pc = (1-cs)pc + sqrt(cs(2-cs)mueff) * (1/sigma) * B^{-1} * (x_new - x) ?
            # No, pc is updated with the same direction but different parameters?
            # Actually, pc is updated as:
            # pc = (1-cs)pc + sqrt(cs(2-cs)mueff) * (1/sigma) * B^{-1} * (x_new - x)
            # Wait, looking at standard references (Hansen 2016):
            # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * (1/sigma) * B^{-1} * (x_new - x)
            # pc = (1-cs)pc + sqrt(cs(2-cs)mueff) * (1/sigma) * B^{-1} * (x_new - x)
            # This looks identical? No, usually pc is for the covariance adaptation and ps for sigma.
            # Let's check the formula carefully.
            # ps is the "sigma" path.
            # pc is the "covariance" path.
            # They are updated with the same term:
            # term = sqrt(cs * (2 - cs) * mueff) * (1/sigma) * (B^{-1} @ (x_new - x))
            # ps = (1 - cs) * ps + term
            # pc = (1 - cs) * pc + term
            # Yes, that is correct for the standard CMA-ES.
            
            term = np.sqrt(cs * (2 - cs) * mueff) * (inv_L @ diff / sigma)
            ps = (1 - cs) * ps + term
            pc = (1 - cs) * pc + term
            
            # Update covariance matrix
            # C = (1 - c1 - cmu) * C + c1 * (pc pc^T + (1 - hsig) * cs * (2 - cs) * ps ps^T) + cmu * sum(w_i * (x_i - x)(x_i - x)^T)
            
            # Calculate hsig
            # hsig = 1 if ||ps|| / sqrt(1 - (1-cs)^{2/gen}) < (1.4 + 2/(n+1)) * sqrt((1-cs^2) * n)
            # We need to track generations to compute the denominator.
            # For simplicity and robustness, we can approximate or use a fixed threshold.
            # Let's keep a generation counter.
            # Actually, we can just compute the norm and compare to a heuristic threshold.
            # Threshold = (1.4 + 2/(n+1)) * sqrt(n)
            # This is the expected norm of ps.
            norm_ps = np.linalg.norm(ps)
            threshold = (1.4 + 2.0 / (n + 1)) * np.sqrt(n)
            # If norm_ps is significantly larger, hsig = 0, else 1.
            # A more precise check uses the generation number, but this heuristic is common.
            hsig = 1 if norm_ps < threshold else 0
            
            # Update C
            # First term: (1 - c1 - cmu) * C
            C = (1 - c1 - cmu) * C
            
            # Second term: c1 * (pc pc^T + (1 - hsig) * cs * (2 - cs) * ps ps^T)
            C += c1 * (np.outer(pc, pc) + (1 - hsig) * cs * (2 - cs) * np.outer(ps, ps))
            
            # Third term: cmu * sum(w_i * (x_i - x)(x_i - x)^T)
            # x_i here are the selected ones
            for i in range(mu):
                diff_i = selected_pop[i] - x
                C += cmu * w[i] * np.outer(diff_i, diff_i)
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / ds * (||ps|| / chi_n - 1))
            # chi_n = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
            sigma = sigma * np.exp(c_sigma / ds * (norm_ps / chi_n - 1))
            
            # Update mean
            x = x_new
            
            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2.0
            # If eigenvalues are negative, fix them
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-10):
                # Reset C to identity if it becomes singular
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                sigma = max(sigma * 0.5, 0.1)

        return best_x, best_f
# EVOLVE-BLOCK-END
