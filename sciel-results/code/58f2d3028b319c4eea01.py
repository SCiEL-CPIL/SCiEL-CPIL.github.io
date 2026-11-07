import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Default lambda for n=5 is 4 + 3*log(5) ~ 7
        lam = max(4 + int(3 * np.log(n)), 1)
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)
        
        # Initialization
        # Start at origin, might be shifted, but CMA-ES is robust
        mean = np.zeros(n)
        sigma = 2.0 # Initial step size, box is [-5, 5]
        C = np.eye(n)
        invsqrtC = np.eye(n)
        
        # Path evolution parameters
        cmac = 1.0 / (n + 3)
        cmu = min(2 / (n + 2) - mueff, 0.5) / mueff
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1) - 1)) + cmu
        
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        best_x = None
        best_f = np.inf
        count = 0
        
        # Pre-calculate constants
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        for gen in range(int(budget / lam) + 1):
            if time.time() > deadline:
                break
            if count >= budget:
                break
            
            # Generate population
            # We need to sample lam points, but we only have budget - count left
            # If remaining budget < lam, we just sample remaining
            remaining = budget - count
            if remaining <= 0:
                break
            
            current_lam = min(lam, remaining)
            
            # Sample
            # z ~ N(0, I)
            Z = np.random.randn(current_lam, n)
            # x = mean + sigma * Z @ invsqrtC.T
            # Note: invsqrtC is such that invsqrtC @ invsqrtC.T = C^-1
            # Actually, we maintain C and its inverse square root.
            # x = mean + sigma * (Z @ invsqrtC.T)
            
            X = mean + sigma * (Z @ invsqrtC.T)
            
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            count += current_lam
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = X[min_idx].copy()
            
            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            # Use top mu points
            weights = w[:min(mu, current_lam)]
            # Adjust weights if current_lam < mu
            if current_lam < mu:
                 # Re-normalize weights for available points
                 weights = weights / weights.sum()
            mueff_curr = 1 / np.sum(weights**2)
            
            mean_new = np.sum(weights[:, None] * X_sorted[:len(weights)], axis=0)
            
            # Update covariance
            # Rank-one update
            # p_sigma = (1 - cmac) * p_sigma + cmac * sqrt(mueff_curr) / sigma * (mean_new - mean) @ invsqrtC
            # Note: (mean_new - mean) is in x-space. We need to transform to z-space for p_sigma update?
            # Standard CMA-ES:
            # p_sigma = (1-cmac)*p_sigma + cmac * sqrt(mueff) * (mean_new - mean) / sigma @ invsqrtC
            # Actually, the term (mean_new - mean) / sigma is the step in the standardized space.
            
            p_sigma = (1 - cmac) * ps + cmac * np.sqrt(mueff_curr) * ((mean_new - mean) / sigma) @ invsqrtC.T
            
            # Update pc
            # pc = (1-cmu)*pc + cmu * sqrt(mueff) * (mean_new - mean) / sigma @ invsqrtC
            # Wait, the update for pc is:
            # pc = (1-cmu)*pc + cmu * sqrt(mueff) * (mean_new - mean) / sigma @ invsqrtC
            # And then C update uses pc @ pc.T and sum of top points deviations.
            
            pc = (1 - cmu) * pc + cmu * np.sqrt(mueff_curr) * ((mean_new - mean) / sigma) @ invsqrtC.T
            
            # Check for restart or stagnation? 
            # Simple check: if sigma becomes too small or too large, restart.
            # For now, just continue.
            
            # Update C
            # C = (1 - cmu * mueff_curr) * C + cmu * (pc @ pc.T + sum(weights_i * (x_i - mean_old) @ invsqrtC.T)^2)
            # The term (x_i - mean_old) @ invsqrtC.T is the standardized deviation.
            
            # Calculate standardized deviations of top points
            # D = X_sorted[:len(weights)] - mean
            # Z_dev = (D / sigma) @ invsqrtC.T
            Z_dev = ((X_sorted[:len(weights)] - mean) / sigma) @ invsqrtC.T
            
            # Rank-mu update part
            # Sum of w_i * z_i @ z_i.T
            rank_mu_part = np.sum(weights[:, None, None] * (Z_dev[:, :, None] * Z_dev[:, None, :]), axis=0)
            
            C = (1 - cmu * mueff_curr) * C + cmu * (pc @ pc.T + rank_mu_part)
            
            # Update sigma
            # sigma = sigma * exp(1/damps * (||ps||/chi_n - 1))
            ps_norm = np.linalg.norm(ps)
            sigma = sigma * np.exp((1.0 / damps) * (ps_norm / chi_n - 1.0))
            
            # Update mean
            mean = mean_new
            
            # Update invsqrtC
            # We need the inverse square root of the new C.
            # Use eigen decomposition. C is symmetric positive definite.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues due to numerical errors
            eigvals = np.clip(eigvals, 1e-12, None)
            invsqrtC = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
            
            # Optional: Regularize C if it becomes ill-conditioned
            cond = eigvals.max() / eigvals.min()
            if cond > 1e12:
                # Restart or regularize
                # For simplicity, let's just clip eigenvalues
                pass

        return best_x, best_f
