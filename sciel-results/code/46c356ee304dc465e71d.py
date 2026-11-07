import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation
    with a final polish phase for improved precision.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Default population size
        lambda_ = 4 + 3 * int(np.floor(np.log2(n)))
        if lambda_ > budget:
            lambda_ = max(2, budget // 2) # Ensure at least some generations if budget is small, but here budget=500
        
        # Weights
        weights = np.log(lambda_ / 2 + 1) - np.log(np.arange(1, lambda_ + 1))
        weights /= np.sum(weights)
        
        # Effective population size
        mu = int(np.floor(lambda_ / 2))
        w_sum = np.sum(weights[:mu])
        mueff = 1 / np.sum(weights[:mu]**2)
        
        # Initial step size
        # Heuristic: start with a moderate step size
        sigma = 2.0 
        
        # Initial mean
        # The problem is on [-5, 5]^5. The minimizer is in [-4, 4]^5.
        # Starting at 0 is a reasonable prior.
        m = np.zeros(n)
        
        # Initial covariance
        C = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Update parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Cholesky factor for sampling (initially identity)
        # We will update C and recompute Cholesky or use eigendecomp if needed.
        # For small n, eigendecomp is cheap and stable.
        
        # Precompute inverse square root of C? Or just use eigendecomp every step?
        # For n=5, eigendecomp is very fast.
        
        def update_chol():
            nonlocal C
            # Symmetrize C
            C = (C + C.T) / 2
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues due to numerical errors
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvals, eigvecs

        # Initial Cholesky/Eig
        eigvals, eigvecs = update_chol()
        sqrt_eigvals = np.sqrt(eigvals)
        
        calls_made = 0
        best_x = m.copy()
        best_f = func(m)
        calls_made += 1
        
        # Reserve some budget for polishing
        polish_budget = max(10, budget // 5)
        cma_budget = budget - polish_budget
        
        while calls_made < cma_budget:
            if time.time() > deadline:
                break
                
            # Sample lambda points
            Y = np.zeros((lambda_, n))
            for i in range(lambda_):
                # Standard normal
                z = np.random.standard_normal(n)
                # Transform using eigenvectors and sqrt eigenvalues
                # x = m + sigma * eigvecs * (sqrt_eigvals * z)
                Y[i] = m + sigma * (eigvecs * sqrt_eigvals) @ z
            
            # Evaluate
            F = np.zeros(lambda_)
            for i in range(lambda_):
                # Clip to box? The task says func(x) clips. 
                # It's better to let the function clip, or clip ourselves to save calls?
                # "func(x) clips a finite length-5 vector to the box and returns a float."
                # So we can pass any vector.
                F[i] = func(Y[i])
                calls_made += 1
                if time.time() > deadline:
                    # If we run out of time mid-generation, we still need to finish the current logic
                    # or break. Breaking here might leave some Y unprocessed.
                    # It's safer to break the outer loop if we exceed deadline.
                    pass
            
            if calls_made >= cma_budget:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
            
            # Update mean
            # m_new = sum_{i=1..mu} w_i * y_{i:lambda}
            m_new = np.dot(weights[:mu], Y_sorted[:mu])
            
            # Update evolution path for sigma
            # p_sigma = (1-c_sigma) p_sigma + c_sigma * sqrt(mueff) * C^{-1/2} * (y_{1:lambda} - m) / sigma
            # C^{-1/2} (y - m) = eigvecs * (z)
            # We have y = m + sigma * C^{1/2} z => z = C^{-1/2} (y-m)/sigma
            # So we can compute z directly from the generation if we stored it, 
            # or recompute. Recomputing is easier here.
            delta = Y_sorted[:mu] - m
            # z_i = C^{-1/2} delta_i / sigma
            # C^{-1/2} = eigvecs * diag(1/sqrt(eigvals)) * eigvecs.T
            # Let's compute C_inv_sqrt_delta
            # delta = sigma * C^{1/2} z => z = (1/sigma) * C^{-1/2} delta
            # C^{-1/2} delta = eigvecs * diag(1/sqrt(eigvals)) * (eigvecs.T @ delta)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma
            for i in range(mu):
                # Compute z for this individual
                delta_i = Y_sorted[i] - m
                # z = eigvecs * ( (1/sqrt_eigvals) * (eigvecs.T @ delta_i) ) / sigma
                proj = eigvecs.T @ delta_i
                z = eigvecs * (proj / sqrt_eigvals)
                p_sigma += c_sigma * np.sqrt(mueff) * weights[i] * z / sigma
                
            # Update sigma
            # sigma *= exp( c_sigma/d_sigma * (||p_sigma||/chi_n - 1) )
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update evolution path for covariance
            # p_c = (1-c_c) p_c + h_sigma * c_c * sqrt(mueff) * C^{-1/2} * (m_new - m)
            # h_sigma = 1 if ||p_sigma||/sqrt(1-(1-c_sigma)^{2gen}) < (1.4 + 2/(n+1)) chi_n else 0
            # gen is not tracked here easily, but we can approximate the threshold.
            # Standard CMA-ES uses a counter. For simplicity and robustness, 
            # we can use a fixed threshold or estimate.
            # Let's use a standard approximation for the threshold.
            threshold = (1.4 + 2 / (n + 1)) * chi_n
            # We need to track the expected norm of p_sigma.
            # ||p_sigma||_expected = chi_n * sqrt(1 - (1-c_sigma)^{2*gen})
            # Since we don't track gen exactly in this loop structure (implicit), 
            # and c_sigma is small, the term approaches chi_n.
            # A common simplification is to check if ||p_sigma|| < threshold.
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < threshold else 0.0
            
            delta_m = m_new - m
            # C^{-1/2} delta_m
            proj_m = eigvecs.T @ delta_m
            z_m = eigvecs * (proj_m / sqrt_eigvals)
            
            p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mueff) * z_m
            
            # Update covariance
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1-h_sigma) c_c (2-c_c) C) + c_mu sum w_i y_i y_i^T
            # where y_i = C^{-1/2} (y_{i:lambda} - m)
            
            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            for i in range(mu):
                delta_i = Y_sorted[i] - m
                proj_i = eigvecs.T @ delta_i
                z_i = eigvecs * (proj_i / sqrt_eigvals)
                C_new += c_mu * weights[i] * np.outer(z_i, z_i)
            
            C = C_new
            
            # Update mean
            m = m_new
            
            # Update eigendecomposition for next step
            eigvals, eigvecs = update_chol()
            sqrt_eigvals = np.sqrt(eigvals)
            
            # Optional: Restart if stagnation? 
            # With budget 500, simple CMA-ES is usually sufficient.
            # If sigma becomes very small, we might stop early.
            if sigma < 1e-8:
                break

        # Polish phase: fine-grained local refinement around the best point
        if calls_made < budget and time.time() < deadline:
            polish_sigma = max(sigma * 0.1, 1e-6)
            # Use a smaller population for polishing
            polish_lambda = max(2, lambda_ // 2)
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                # Sample polish points
                Y_polish = np.zeros((polish_lambda, n))
                for i in range(polish_lambda):
                    z = np.random.standard_normal(n)
                    Y_polish[i] = best_x + polish_sigma * z
                
                # Evaluate
                F_polish = np.zeros(polish_lambda)
                for i in range(polish_lambda):
                    F_polish[i] = func(Y_polish[i])
                    calls_made += 1
                    if time.time() > deadline:
                        break
                
                if calls_made >= budget:
                    break
                
                # Update best
                idx_best = np.argmin(F_polish)
                if F_polish[idx_best] < best_f:
                    best_f = F_polish[idx_best]
                    best_x = Y_polish[idx_best].copy()
                    # Slightly reduce sigma for next polish iteration
                    polish_sigma *= 0.8
                else:
                    # If no improvement, shrink sigma more aggressively
                    polish_sigma *= 0.5
                
                if polish_sigma < 1e-10:
                    break

        return best_x, best_f
