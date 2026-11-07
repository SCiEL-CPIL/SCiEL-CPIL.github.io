"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style Schwefel function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        # Weights for the best mu individuals
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial step size
        # The box is [-5, 5]. A good initial sigma is roughly the side length / 2 or less.
        # For Schwefel, global minimum is often far from 0 if shifted, but we start at 0.
        # Let's start with sigma = 2.0 to cover a good portion of the space.
        sigma = 2.0
        
        # Mean vector
        m = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Eigendecomposition cache for efficiency
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Helper for inverse sqrt of C
        def invsqrtC(C, B, D):
            # C = B @ diag(D^2) @ B.T
            # invsqrt(C) = B @ diag(1/D) @ B.T
            return B @ (np.diag(1.0 / D) @ B.T)
        
        best_f = np.inf
        best_x = np.zeros(dim)
        calls_made = 0
        
        # Pre-compute inverse sqrt of initial C (identity)
        # In CMA-ES, we usually maintain B and D from eigendecomposition
        
        while calls_made + lam <= budget:
            if time.time() > deadline:
                break
            
            # Sample lambda points
            # y = B @ (D * z)
            # z ~ N(0, I)
            Z = np.random.standard_normal((lam, dim))
            # Update B and D if needed? 
            # For simplicity and speed in 5D, we can just use the current C directly 
            # via Cholesky or just sample from N(0, C) if we don't maintain B, D strictly.
            # However, standard CMA-ES uses B and D. Let's do a simple eigendecomposition update each step.
            # In 5D, this is cheap.
            
            # Eigendecomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Sample
            Y = Z @ np.diag(D) @ B.T  # (lam, dim)
            X = m[None, :] + sigma * Y  # (lam, dim)
            
            # Clip to box
            X = np.clip(X, -5.0, 5.0)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lam
            
            # Track best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            m_new = X_sorted[:mu] @ weights
            
            # Update path p_sigma
            # h_sigma = 1 if ||p_sigma|| < (1 + c_sigma) * sqrt(1 - (1 - c_sigma)^{2/(calls+1)}) * sqrt(mueff) * ||C^{-1/2}||_F ... 
            # Simplified: use standard condition
            # norm_p_sigma = ||p_sigma_new||
            # We need C^{-1/2}
            C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ (m_new - m) / sigma
            
            # Threshold for h_sigma
            # Theoretical value is complex, use a heuristic or standard approximation
            # Standard: h_sigma = 1 if ||p_sigma|| < (1 + c_sigma) * sqrt(1 - (1 - c_sigma)^{2*(calls+1)}) * sqrt(mueff) * sqrt(trace(C) / dim)
            # Let's use a simpler check: if p_sigma is not too large relative to expected
            # Actually, let's just compute the norm and compare to a threshold derived from mueff
            # For robustness, let's use the standard update and a simple h_sigma = 1.0 if norm < 3*sqrt(mueff) else 0
            # A better approximation for h_sigma:
            # E[||z||^2] = dim. E[||C^{-1/2} z||^2] = trace(C^{-1}C) = dim? No.
            # Let's stick to a robust implementation.
            
            # Update path p_c
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1 + c_sigma) * np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + lam))) * np.sqrt(mueff) * np.sqrt(np.trace(C) / dim) else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - m) / sigma
            
            # Update covariance matrix
            # Delta_h = (2 - h_sigma) * c_c * (1 - c_c)
            delta_h = (2 - h_sigma) * c_c * (1 - c_c)
            
            # Sum of outer products of best mu
            # H = (1/mueff) * sum_{i=1}^mu w_i * (y_i/sigma) (y_i/sigma)^T
            # y_i = x_i - m_old
            Y_best = X_sorted[:mu] - m
            H = (Y_best.T / sigma) @ np.diag(weights / mueff) @ (Y_best / sigma)
            
            C = (1 + c_1 + c_mu - c_1 - c_c * (1 - c_c) + c_c * (1 - c_c) - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) + delta_h * C - np.eye(dim)) \
                + c_mu * H
            
            # Symmetrize to prevent numerical drift
            C = (C + C.T) / 2
            
            # Update sigma
            # chi_n = E[||N(0, I)||] = sqrt(dim) * (1 - 1/(4dim) + 1/(32dim^2))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (32 * dim**2))
            
            # ||p_sigma|| / chi_n
            # Use a moving average? No, standard CMA-ES uses:
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            # d_sigma = 1 + 2*max(0, sqrt((mueff-1)/(dim+1) - 1)) + c_sigma
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1)) + c_sigma
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            m = m_new
            
        return best_x, best_f
# EVOLVE-BLOCK-END
