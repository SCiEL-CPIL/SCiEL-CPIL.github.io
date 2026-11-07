"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style Weierstrass function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For dim=5, lambda is typically 4 + floor(3*ln(5)) = 4+5=9 or similar
        # Let's use a standard lambda
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        # Initial parameters
        x0 = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Update rates
        cc = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        cs = (1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1)) / (dim + 2 + mu / 2)
        c1 = 2 / ((dim + 1.3) ** 2 + mu)
        cmu = min(1 - c1, 2 * (mu_eff - 2) / ((dim + 2) ** 2 + mu_eff)) if (mu_eff := 2 * mu / lam) else 0
        damps = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + cs
        
        # Precompute weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Recalculate cmu with correct mu_eff
        cmu = min(1 - c1, 2 * (mu_eff - 2) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial population
        rng = np.random.default_rng(42)
        
        # Generate initial population
        def sample_population(x, sigma, C):
            # Whitening matrix from C
            # Use Cholesky decomposition for efficiency
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            z = rng.standard_normal((lam, dim))
            # Transform z
            pop = x + sigma * (z @ L.T)
            return pop, z
        
        # Initial evaluation
        pop, z = sample_population(x0, sigma, C)
        fvals = np.array([func(x) for x in pop])
        
        # Sort by fitness
        indices = np.argsort(fvals)
        pop = pop[indices]
        fvals = fvals[indices]
        z = z[indices]
        
        best_x = pop[0]
        best_f = fvals[0]
        
        # Track best so far
        global_best_x = best_x.copy()
        global_best_f = best_f
        
        calls_made = lam
        
        for gen in range(1, budget // lam + 1):
            # Update mean
            x_mean = np.sum(w[:, np.newaxis] * pop[:mu], axis=0)
            
            # Update step size control
            x_mean_diff = x_mean - x0
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * L.T @ x_mean_diff
            
            # Check if ps is too large
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                # Update covariance matrix
                delta_sig = np.linalg.norm(ps) / np.sqrt(dim) - 1
                sigma *= np.exp(cs / damps * delta_sig)
                
                # Update C
                rank_one = np.outer(x_mean_diff, x_mean_diff)
                rank_mu = np.sum(w[:mu, np.newaxis, np.newaxis] * np.outer(pop[:mu] - x0, pop[:mu] - x0), axis=0) / sigma ** 2
                
                C = (1 + c1 * cc * (2 - cc) - c1 - cmu) * C + c1 * rank_one / sigma ** 2 + cmu * rank_mu
                C = (C + C.T) / 2  # Symmetrize
                
                # Update pc
                pc = (1 - cc) * pc + cc * x_mean_diff / sigma
            
            x0 = x_mean
            
            # Sample new population
            pop, z = sample_population(x0, sigma, C)
            
            # Evaluate
            for i in range(lam):
                if calls_made >= budget:
                    break
                fvals[i] = func(pop[i])
                calls_made += 1
                if fvals[i] < global_best_f:
                    global_best_f = fvals[i]
                    global_best_x = pop[i].copy()
            
            if calls_made >= budget:
                break
            
            # Sort by fitness
            indices = np.argsort(fvals)
            pop = pop[indices]
            fvals = fvals[indices]
            z = z[indices]
            
            # Update best
            if fvals[0] < global_best_f:
                global_best_f = fvals[0]
                global_best_x = pop[0].copy()
            
            # Check for stagnation
            if gen > 50 and fvals[0] == fvals[-1]:
                break
        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
