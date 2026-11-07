"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-Based CMA-ES for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 2 / 3) / (dim + 2)) ** 2)
        
        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Initial population
        rng = np.random.default_rng(42)
        pop = x + sigma * rng.standard_normal((lam, dim))
        pop = np.clip(pop, -5, 5)
        
        # Evaluate initial population
        values = np.array([func(p) for p in pop])
        calls_made = lam
        
        # Track best
        best_val = np.min(values)
        best_x = pop[np.argmin(values)]
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        while calls_made + lam <= budget and time.time() < deadline:
            # Sort by fitness
            order = np.argsort(values)
            y_sorted = pop[order]
            f_sorted = values[order]
            
            # Update mean
            x_new = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            
            # Inverse Cholesky transpose for ps update
            # We need Cinv^{1/2} which is L^{-T} if C = L L^T
            try:
                L_inv_T = np.linalg.inv(L).T
            except np.linalg.LinAlgError:
                L_inv_T = np.eye(dim)
            
            # Generation number for hsigma
            gen = calls_made // lam + 1
            
            # Update step size control vector ps
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) * Cinv^{1/2} * (x_new - x)/sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L_inv_T @ ((x_new - x) / sigma))
            
            # Update evolution path pc
            # hsigma condition
            norm_ps = np.linalg.norm(ps)
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            # The standard condition uses the expected norm of ps
            # E[||ps||] = sqrt(1 - (1-cs)^(2t/(dim+1))) * chiN
            # So ratio = norm_ps / (sqrt(1 - (1-cs)^(2t/(dim+1))) * chiN)
            # And we compare to (1 + 2/(dim+1)) * sqrt(dim + 1 - 2) ?
            # Actually, the threshold is often:
            # (1 + 2/(dim+1)) * sqrt(dim + 1 - 2)
            # Let's use the standard formula from Hansen's CMA-ES tutorial
            denom = np.sqrt(1 - (1 - c_sigma) ** (2 * gen / (dim + 1))) * chiN
            if denom > 1e-10:
                hsigma = (norm_ps / denom) < (1 + 2 / (dim + 1)) * np.sqrt(dim + 1 - 2)
            else:
                hsigma = True
            
            pc = (1 - c_c) * pc + hsigma * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x) / sigma
            
            # Update covariance matrix C
            mu_sigma = c_1 + c_mu - 1
            coeff_C = 1 - c_1 - c_mu + mu_sigma
            C = coeff_C * C
            C += c_1 * (np.outer(pc, pc) - (1 - hsigma) * (1 - c_c) * np.eye(dim))
            
            # Rank-mu update
            diff = y_sorted[:mu] - x
            C += c_mu * (diff.T @ (weights[:, None] * diff))
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chiN - 1))
            
            # Update mean
            x = x_new
            
            # Generate new population
            pop = x + sigma * (L @ rng.standard_normal((lam, dim)).T).T
            pop = np.clip(pop, -5, 5)
            
            # Evaluate
            values = np.array([func(p) for p in pop])
            calls_made += lam
            
            # Update best
            current_best_idx = np.argmin(values)
            if values[current_best_idx] < best_val:
                best_val = values[current_best_idx]
                best_x = pop[current_best_idx]

        return best_x, best_val
# EVOLVE-BLOCK-END
