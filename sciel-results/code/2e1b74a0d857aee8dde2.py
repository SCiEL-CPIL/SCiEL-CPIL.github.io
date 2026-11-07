import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        # Step size
        sigma = 2.0
        # Initial mean
        x_mean = np.zeros(n)
        # Covariance matrix
        C = np.eye(n)
        # Inverse square root of C for sampling
        # We can use eigenvalue decomposition or Cholesky, but for small n, 
        # updating C and recomputing Cholesky each step is fine.
        
        # Control parameters
        mu = lambda_ // 2  # Number of recombination points
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Evaluation budget tracking
        evals_left = self.budget
        
        while evals_left > 0:
            # Generate population
            # Compute Cholesky factor of C
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite due to numerical issues, reset or fix
                C = np.eye(n)
                B = np.eye(n)
            
            # Sample lambda points
            z = np.random.randn(lambda_, n)
            # y = sigma * B * z^T
            y = sigma * (z @ B.T)
            # x = x_mean + y
            x = x_mean[None, :] + y
            
            # Evaluate fitness
            # We can evaluate all lambda points. 
            # Check if we have enough budget
            if evals_left < lambda_:
                # If not enough for a full generation, evaluate remaining one by one
                # This is a corner case, but good for safety
                for i in range(evals_left):
                    f = func(x[i])
                    if f < best_f:
                        best_f = f
                        best_x = x[i].copy()
                    evals_left -= 1
                break
            
            f = np.array([func(x[i]) for i in range(lambda_)])
            evals_left -= lambda_
            
            # Sort by fitness
            idx = np.argsort(f)
            x_sorted = x[idx]
            f_sorted = f[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
            
            # Recombination
            # Weighted sum of the best mu points
            x_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update step size
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B.T @ (x_new - x_mean)
            # Note: The formula usually involves the inverse of the old C.
            # Standard CMA-ES update:
            # p_sigma = (1 - c_sigma) p_sigma + h_sigma * sqrt(c_sigma (2 - c_sigma) mu_eff) * C^{-1/2} (x_new - x_mean) / sigma
            # h_sigma is a heuristic to prevent divergence.
            
            # Calculate C^{-1/2} (x_new - x_mean) / sigma
            # This is equivalent to B^{-1} @ ((x_new - x_mean) / sigma)
            # Since B is Cholesky factor (lower triangular), B @ B.T = C
            # C^{-1/2} = B^{-T} @ B^{-1} ? No, usually we use the matrix B from the Cholesky of C.
            # The step direction is d = (x_new - x_mean) / sigma
            # We need C^{-1/2} d.
            # Let's use the property that if C = B B^T, then C^{-1/2} is not simply B^{-1}.
            # However, in the standard algorithm, we often keep the transformation matrix B (Cholesky factor)
            # and update it.
            # A simpler implementation for small n:
            # Compute C^{-1/2} via eigenvalue decomposition or just solve linear systems.
            
            # Let's stick to a robust implementation.
            # p_sigma update requires C^{-1/2} (x_new - x_mean) / sigma
            # Let d = (x_new - x_mean) / sigma
            # We need v such that C v = d. Then C^{-1/2} d is not v.
            # Actually, the standard update is:
            # p_sigma = (1 - c_sigma) p_sigma + h_sigma * sqrt(c_sigma (2 - c_sigma) mu_eff) * C^{-1/2} (x_new - x_mean) / sigma
            
            # To compute C^{-1/2} d, we can do:
            # 1. Solve C u = d
            # 2. Compute eigenvalues/vectors of C, or use Cholesky.
            # If C = L L^T, then C^{-1/2} d = L^{-T} L^{-1} d ? No.
            # C^{-1/2} = (L L^T)^{-1/2}.
            # It is easier to maintain the matrix B such that C = B B^T.
            # Then C^{-1/2} d = B^{-T} (B^{-1} d) ? No.
            # Let's use the formula: C^{-1/2} = B^{-1} (B B^T)^{-1/2} B^{-T} ... this is getting complicated.
            
            # Alternative: Use the update rule that directly updates C and uses a helper for the p_sigma update.
            # Many implementations use:
            # p_sigma = (1 - c_sigma) p_sigma + h_sigma * sqrt(c_sigma (2 - c_sigma) mu_eff) * (x_new - x_mean) / sigma
            # This is an approximation if C is not identity. The correct term is C^{-1/2} (x_new - x_mean) / sigma.
            # For a robust and simple implementation, we can compute the matrix square root inverse.
            
            # Let's compute C^{-1/2} using eigen decomposition for n=5. It's fast enough.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            d_sigma_vec = C_inv_sqrt @ ((x_new - x_mean) / sigma)
            
            # h_sigma
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2 gen}) < (1 + 2/(n+1)) * E||N(0,I)||
            # This is complex. A common simplification is to check if the norm is too large.
            # Or just use the standard threshold.
            # Let's implement the standard h_sigma.
            # We need to track generations or use an approximation.
            # For simplicity and robustness in a short budget, we can omit h_sigma or set it to 1 if not diverging.
            # Actually, let's include a simple check.
            # If ||p_sigma|| > (1 + 2/(n+1)) * sqrt(n), set h_sigma = 0? No, that's for preventing explosion.
            # Let's just use h_sigma = 1 for now, as CMA-ES is robust.
            h_sigma = 1.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_sigma_vec
            
            # Update p_c
            # p_c = (1 - c_c) p_c + h_sigma * sqrt(c_c (2 - c_c) mu_eff) * (x_new - x_mean) / sigma
            # Note: This is the same vector d_sigma_vec * sigma? No.
            # The term is (x_new - x_mean) / sigma.
            # Wait, the p_c update uses the un-normalized step?
            # Standard: p_c = (1 - c_c) p_c + h_sigma * sqrt(c_c (2 - c_c) mu_eff) * (x_new - x_mean) / sigma
            # Yes, it uses the normalized step.
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * ((x_new - x_mean) / sigma)
            
            # Update C
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu sum w_i (y_i y_i^T)
            # where y_i = (x_i - x_mean) / sigma
            
            # Calculate the rank-mu update
            # y_sorted = (x_sorted[:mu] - x_mean) / sigma
            y_sorted = (x_sorted[:mu] - x_mean[None, :]) / sigma
            # C_mu = sum w_i y_i y_i^T
            C_mu = np.sum(weights[:, None, None] * y_sorted[:, :, None] * y_sorted[:, None, :], axis=0)
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * C_mu
            
            # Update mean
            x_mean = x_new
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / E||N(0,I)|| - 1))
            # E||N(0,I)|| = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
            E_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E_norm - 1))
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-12, 1e12)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Check eigenvalues
            eigvals_check = np.linalg.eigvalsh(C)
            if np.any(eigvals_check < 1e-12):
                # Reset C if it becomes singular
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                sigma = 2.0

        return best_x, best_f
