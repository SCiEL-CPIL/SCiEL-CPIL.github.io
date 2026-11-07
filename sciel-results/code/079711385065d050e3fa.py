"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Count function calls
        call_count = [0]
        
        def counted_func(x):
            call_count[0] += 1
            # Clip to box to ensure validity
            x_clipped = np.clip(x, -5, 5)
            return float(func(x_clipped))
        
        # CMA-ES Parameters
        lam = max(4 + int(3 * np.log(dim)), 4)
        
        # Initial mean
        m = np.zeros(dim)
        
        # Initial step size
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Weights
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Update rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        
        # Damping
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Best solution
        best_x = m.copy()
        best_f = counted_func(m)
        
        # Pre-compute C^{-1/2} approximation for ps update (using Cholesky)
        # We will update L in the loop
        
        while call_count[0] < budget:
            remaining = budget - call_count[0]
            current_lam = min(lam, remaining)
            
            if current_lam <= 0:
                break
            
            # Sample offspring
            # Get Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-10 * np.eye(dim)
                L = np.linalg.cholesky(C)
            
            # Standard normal samples
            z = np.random.randn(current_lam, dim)
            
            # Transform
            # x_i = m + sigma * L @ z_i^T
            # Since z is (lam, dim), we want each row z_i to be multiplied by L.
            # x_i = m + sigma * (L @ z_i.T).T = m + sigma * z_i @ L.T
            offspring = m + sigma * (z @ L.T)
            
            # Evaluate
            values = np.array([counted_func(x) for x in offspring])
            
            # Update best
            idx_best = np.argmin(values)
            if values[idx_best] < best_f:
                best_f = values[idx_best]
                best_x = offspring[idx_best].copy()
            
            # Sort
            order = np.argsort(values)
            y = offspring[order]
            f = values[order]
            
            # Save old mean
            m_old = m.copy()
            
            # Update mean
            # Note: w is defined for mu, but we use current_lam. 
            # If current_lam < mu, we should renormalize weights or just use the available ones.
            # Standard CMA-ES assumes fixed lam. Here we adapt lam to budget.
            # We must renormalize weights for the active offspring.
            w_active = w[:current_lam]
            w_active = w_active / np.sum(w_active)
            
            m = np.sum(w_active * y[:current_lam], axis=0)
            
            # Update evolution path ps
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * C^{-1/2} * (m - m_old) / sigma
            diff = (m - m_old) / sigma
            try:
                # Solve L v = diff
                # L is lower triangular
                v = np.linalg.solve(L, diff)
            except np.linalg.LinAlgError:
                v = diff # Fallback
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * v
            
            # Update sigma
            # Standard CMA-ES sigma update
            chi0 = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            ps_norm = np.linalg.norm(ps)
            
            # The term (1 - (1 - cs) ** (2 * t)) approximates the variance of ps
            # t is generation number. We use call_count // lam as an approximation.
            gen = call_count[0] // lam
            ps_norm_scaled = ps_norm / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
            
            sigma *= np.exp(cs / damps * (ps_norm_scaled / chi0 - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Calculate hsig
            hsig = 1 if ps_norm_scaled < (1.4 + 2 / (dim + 1)) * chi0 else 0
            
            # Update pc
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m - m_old) / sigma
            
            # Update covariance matrix C
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - hsig) * C) + cmu * H
            # H = sum w_i (y_i - m) (y_i - m).T
            
            y_w = y[:current_lam] - m
            H = y_w.T @ (w_active[:, None] * y_w)
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * C) + cmu * H
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # If sigma is very small, we might be converged
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
