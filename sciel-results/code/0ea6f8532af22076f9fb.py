import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 300.0  # Safety margin for 330s limit
        
        # CMA-ES Parameters
        # For dim=5, lambda = 4 + floor(3 * ln(5)) = 4 + 4 = 8? 
        # Standard: lambda = 4 + floor(3 * ln(d))
        # ln(5) ~ 1.6, 3*1.6 = 4.8, floor = 4. lambda = 8.
        # Let's use lambda = 10 or 12 for better exploration in early stages.
        lambda_ = 12 
        
        # Weights
        mu = lambda_ // 2
        # Rank-mu update weights
        w = np.log(mu + 1) - np.arange(mu)
        w /= w.sum()
        # Rank-1 update weight
        mueff = 1 / sum(w**2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial step size
        # The box is [-5, 5]. The optimum is in [-4, 4].
        # A good initial sigma is roughly half the box width? Or smaller.
        # Let's start with sigma = 2.0 to cover the space quickly.
        sigma = 2.0
        
        # Initial mean: random point in [-2, 2] to be central
        rng = np.random.default_rng(42)
        m = rng.uniform(-2, 2, dim)
        
        # Initial covariance: Identity
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        calls = 0
        best_x = m.copy()
        best_f = func(best_x)
        calls += 1
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Precompute some constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        while calls + lambda_ <= budget:
            if time.time() > deadline:
                break
                
            # Generate samples
            # We need to update C_sqrt if C changes significantly, but for simplicity 
            # in a compact implementation, we can use Cholesky decomposition every step 
            # or approximate. For dim=5, Cholesky is very fast.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset C to identity
                C = np.eye(dim)
                L = np.eye(dim)
            
            samples = []
            fvals = []
            
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                x = m + sigma * L @ z
                # Clip to box? The problem says func clips, but we can clip ourselves 
                # to save potential numerical issues, though func handles it.
                # It's better to let func clip as per contract, but ensure x is finite.
                fx = func(x)
                calls += 1
                if calls > budget:
                    break
                samples.append(x)
                fvals.append(fx)
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    
            if len(samples) < lambda_:
                break
                
            samples = np.array(samples)
            fvals = np.array(fvals)
            
            # Sort by fitness
            idx = np.argsort(fvals)
            samples_sorted = samples[idx]
            fvals_sorted = fvals[idx]
            
            # Update mean
            # m_new = sum(w_i * x_{i:i+mu})
            m_new = np.sum(w[:, None] * samples_sorted[:mu], axis=0)
            
            # Update step size
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (m_new - m) @ C_sqrt
            # We need C_sqrt inverse? No, the formula is:
            # p_sigma = (1-c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma)mueff) * (1/sigma) * C^{-1/2} (m_new - m)
            # Wait, standard CMA-ES uses C^{1/2} in the generation, so the update is:
            # p_sigma = (1-c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma)mueff) * C^{-1/2} (m_new - m) / sigma
            
            # We have L such that L L^T = C. So C^{1/2} is not exactly L, but L is a square root.
            # To get C^{-1/2} (m_new - m), we solve L L^T y = (m_new - m) / sigma
            # Or simpler: The standard update is:
            # p_sigma = (1-c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma)mueff) * (m_new - m) / (sigma * ||C^{1/2}||?)
            
            # Let's use the standard formulation from Hansen's paper:
            # p_sigma = (1-c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma)mueff) * C^{-1/2} (m_new - m) / sigma
            
            # Compute C^{-1/2} (m_new - m)
            # Since L @ L.T = C, C^{-1} = L^{-T} @ L^{-1}
            # C^{-1/2} is the matrix square root of C^{-1}.
            # We can solve L.T @ u = (m_new - m) / sigma, then C^{-1/2} (m_new-m)/sigma = L^{-1} u ?
            # Actually, if x = C^{1/2} z, then z = C^{-1/2} x.
            # We generated x = m + sigma L z. So z = L^{-1} (x - m) / sigma.
            # The term in p_sigma update is essentially the average of the z-vectors of the selected points?
            # No, the standard update uses the difference of means.
            
            # Let's stick to the robust implementation:
            # diff = (m_new - m) / sigma
            # z_diff = np.linalg.solve(L.T, np.linalg.solve(L, diff)) # This is C^{-1} diff? No.
            # L L^T = C. 
            # We want C^{-1/2} diff.
            # Let A = C^{-1/2}. Then A A = C^{-1}.
            # We can compute C^{-1} via cholesky of C.
            # C_inv = np.linalg.inv(C) is risky.
            
            # Alternative: Use the fact that we have L.
            # p_sigma update:
            # p_sigma = (1 - c_sigma) * p_sigma
            # coeff = np.sqrt(c_sigma * (2 - c_sigma) * mueff)
            # # Solve L.T y = (m_new - m) / sigma
            # y = np.linalg.solve(L.T, (m_new - m) / sigma)
            # # Then z = L^{-1} y? No.
            # # If L L^T = C, then C^{-1/2} is not simply L^{-1}.
            # # However, the standard CMA-ES code often computes:
            # # p_sigma += coeff * C_inv_sqrt @ (m_new - m) / sigma
            # # Where C_inv_sqrt is computed from the eigen-decomposition or similar.
            
            # For dim=5, we can just compute the matrix square root inverse directly.
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12) # Ensure positive
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + coeff * (C_inv_sqrt @ (m_new - m) / sigma)
            
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) < (1 + 2 / (dim + 1)) * chi_n else 0.0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - m) / sigma
            
            # Update C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu sum w_i (y_i y_i^T)
            # where y_i = C^{-1/2} (x_i - m) / sigma
            
            # Compute y_i for the mu best points
            Y = np.zeros((mu, dim))
            for i in range(mu):
                Y[i] = C_inv_sqrt @ (samples_sorted[i] - m) / sigma
            
            # Rank-1 part
            rank1 = np.outer(p_c, p_c)
            if h_sigma < 1.0:
                rank1 += (1 - h_sigma) * c_c * (2 - c_c) * C
            
            # Rank-mu part
            rank_mu = np.sum(w[:, None, None] * Y[:, :, None] * Y[:, None, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rank_mu
            
            # Update sigma
            sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            m = m_new
            
            # Optional: If sigma is too small, stop
            if sigma < 1e-12:
                break

        return best_x, best_f
