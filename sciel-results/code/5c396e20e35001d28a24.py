import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Lambda: population size
        lam = 4 + int(3 * np.log(n))
        if lam < 4: lam = 4
        
        # Weights for the top mu individuals
        mu = lam // 2
        # Weights: w_i proportional to log(lam - i + 1) - log(lam - mu)
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(lam - i + 1) - np.log(lam - mu)
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        # Start at 0,0,0,0,0. The shift is unknown, but 0 is a reasonable center for [-5,5].
        # Or we could do a small random initialization.
        # The task says minimizer is in [-4,4]^5.
        # Let's start at 0.
        m = np.zeros(n)
        sigma = 2.0 # Initial step size. Range is 10, so 2 is reasonable.
        C = np.eye(n)
        C_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Pre-compute Hessian of C? No, we update C directly.
        # To maintain symmetry and PD, we can use the eigen decomposition trick 
        # or just enforce symmetry and check eigenvalues.
        # For n=5, it's cheap to just update and symmetrize.
        
        best_x = m.copy()
        best_f = func(best_x)
        calls = 1
        
        while calls < budget:
            # Check time
            if time.time() - start_time > 290:
                break
                
            # Generate population
            Y = np.zeros((lam, n))
            for i in range(lam):
                # Standard normal
                z = np.random.randn(n)
                # Apply transformation: C^(1/2)
                # We need C^(1/2). We can compute it via eigendecomposition of C.
                # To save time, we can maintain the Cholesky factor or Eigen decomposition.
                # For simplicity and robustness in a short script, let's compute eigendecomposition each step?
                # That might be slow. 
                # Alternative: Use the fact that we update C, and we can maintain the eigen-decomp.
                # Or, simpler: Just use np.linalg.eigh(C) each iteration. n=5, this is very fast.
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues to be positive
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                
                Y[i] = m + sigma * (C_sqrt @ z)
            
            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                F[i] = func(Y[i])
                calls += 1
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
                
            # Sort
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
                
            # Update Mean
            m_new = m.copy()
            for i in range(mu):
                m_new += weights[i] * Y_sorted[i]
            m = m_new
            
            # Update Paths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (m_new - m) / sigma
            # Wait, the formula is: p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2-c_sigma) mu_eff) / sigma * C^{-1/2} (m_new - m)
            # We have C_sqrt, so C_inv_sqrt = C_sqrt^{-1}
            # C_inv_sqrt = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
            
            # Let's re-calculate C_inv_sqrt from the eigendecomposition we just did? 
            # Or store it.
            # Actually, we can just compute it.
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (m_new - m) / sigma)
            
            # Check if p_sigma is too large
            chi_n = np.sqrt(n) * (1 + 1 / (2 * n) - 1 / (8 * n**2))
            if np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (n + 1)) * np.sqrt((1 - (1 - c_sigma)**(2 * (budget - calls)))):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
            # p_c update
            h_sigma = 1 if np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma)**(2 * (budget - calls))) else 0
            # Note: The condition for h_sigma depends on the expected norm.
            # A simpler approximation often used:
            h_sigma = 1 if np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (n + 1)) else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (m_new - m) / sigma
            
            # Update Covariance
            # Rank-1 update
            y_c = (m_new - m) / sigma
            C = (1 + c_1 * (h_sigma * (2 - c_c) - c_c) - c_1) * C + c_1 * np.outer(y_c, y_c)
            
            # Rank-mu update
            # C = (1 + c_mu * (2 - c_c) - c_1 - c_mu) * C + c_mu * sum(w_i * y_i * y_i^T)
            # Wait, the standard update is:
            # C <- (1 - c_1 - c_mu) C + c_1 y_c y_c^T + c_mu sum w_i y_i y_i^T
            # Let's stick to the standard formula.
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(y_c, y_c)
            for i in range(mu):
                y_i = (Y_sorted[i] - m) / sigma # Note: m is the OLD mean? 
                # The formula usually uses the difference from the OLD mean m.
                # m_new was used for p updates. The covariance update uses the step vectors from the old mean.
                C += c_mu * weights[i] * np.outer(y_i, y_i)
                
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                C = eigvecs @ np.diag(np.clip(eigvals, 1e-12, None)) @ eigvecs.T
            else:
                # Re-orthonormalize if necessary? Eigh is stable.
                pass
                
            # Update C_inv? We don't explicitly need C_inv for the next step if we re-compute C_sqrt.
            # But we need C_inv_sqrt for p updates. We computed it above.
            
            # Safety: if sigma becomes too small or too large
            if sigma < 1e-12:
                break
            if sigma > 1e6:
                break

        return best_x, best_f
