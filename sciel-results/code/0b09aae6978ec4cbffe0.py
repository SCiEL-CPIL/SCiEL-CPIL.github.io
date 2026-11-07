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
        best_f = np.inf
        
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
            m = np.sum(w[:current_lam] * y[:current_lam], axis=0)
            
            # Update evolution path ps
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * C^{-1/2} * (m - m_old) / sigma
            # C^{-1/2} is approximated by inverse of L (upper triangular)
            # Actually, C^{1/2} = L. So C^{-1/2} = L^{-1}.
            # We need to solve L v = (m - m_old) / sigma
            diff = (m - m_old) / sigma
            try:
                # Solve L v = diff
                # L is lower triangular
                v = np.linalg.solve(L, diff)
            except np.linalg.LinAlgError:
                v = diff # Fallback
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * v
            
            # Check if ps is too large
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (call_count[0] // lam + 1))) > (1.4 + 2 / (dim + 1)) * np.sqrt(dim + 1):
                # Step size increase
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (call_count[0] // lam + 1))) - np.sqrt(dim + 1)))
            
            # Update covariance matrix C
            # C = (1 - mu_1 - c1 + cmu) * C + c1 * (ps ps^T + (1 - delta_sigma) * 1_{||ps||^2/chi0^2 > 1} * ps ps^T) + cmu * sum w_i (y_i - m)(y_i - m)^T
            # delta_sigma = 1 - (1 - cs) ** 2
            delta_sigma = 1 - (1 - cs) ** 2
            
            # Term 1: (1 - mu_1 - c1 + cmu) * C
            # mu_1 = sum w_i^2
            mu_1 = np.sum(w[:current_lam]**2)
            
            # Term 2: c1 * (ps ps^T + ...)
            ps_norm_sq = np.dot(ps, ps)
            chi0 = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            delta_sigma_term = 1 if ps_norm_sq / chi0**2 > 1 else 0
            
            # Term 3: cmu * sum w_i (y_i - m)(y_i - m)^T
            # y is sorted, use top mu
            y_w = y[:current_lam] - m
            # Outer product sum
            # cmu * (y_w.T @ (w[:current_lam, None] * y_w))
            cov_sum = y_w.T @ (w[:current_lam, None] * y_w)
            
            C = (1 - mu_1 - c1 + cmu) * C + c1 * (np.outer(ps, ps) + delta_sigma_term * np.outer(ps, ps)) + cmu * cov_sum
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Step size update for sigma
            # sigma = sigma * exp(cs / damps * (||ps|| / chi0 - 1))
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp(cs / damps * (ps_norm / chi0 - 1))
            
            # Ensure sigma is not too small or too large
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update pc
            # pc = (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * (m - m_old) / sigma
            hsig = 1 if ps_norm / np.sqrt(1 - (1 - cs) ** (2 * (call_count[0] // lam + 1))) < (1.4 + 2 / (dim + 1)) * chi0 else 0
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m - m_old) / sigma
            
            # Update C with pc term? 
            # The standard formula above already includes the main adaptation.
            # The pc term is often added as:
            # C += (c1 + cmu) * (1/delta_sigma - 1 + 2) * (pc pc^T - C) ? 
            # Actually, the formula used above is the standard one from Hansen's paper.
            
            # If sigma is very small, we might be converged
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
