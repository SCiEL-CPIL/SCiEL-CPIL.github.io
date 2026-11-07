"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        
        # Initial parameters
        # Start in the center of the box, which is often a good heuristic for symmetric domains
        mean = np.zeros(n)
        # Initial step size (sigma). 2.0 is a common starting value for [-5, 5]
        sigma = 2.0
        # Covariance matrix
        C = np.eye(n)
        
        # CMA-ES parameters
        # Population size
        # Standard choice: lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(n)) if n > 1 else 2
        # Ensure lam is at least 2 and odd for some implementations, but even is fine for standard
        # Let's use a fixed population size that fits the budget well.
        # Budget 500, n=5. 
        # If lam=10, iterations = 50.
        # If lam=20, iterations = 25.
        # CMA-ES converges fast. Let's try lam=12 (4 + 3*log(5) ~ 4+4.8 = 8.8 -> 9? No, floor is 8. 4+8=12).
        
        # Weights for the mean update
        mu = lam // 2  # Number of points contributing to the mean update
        # Weights: w_i proportional to log(mu+1) - log(i)
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        # Step-size adaptation
        cs = (mueff + 2) / (n + mueff + 5)
        # Covariance matrix adaptation
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        
        # Damping for the step size
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        # Cumulation path for step size
        ps = np.zeros(n)
        # Cumulation path for covariance
        pc = np.zeros(n)
        
        # Precompute some values
        # Eigenvalue decomposition of C for sampling
        # We will update C, so we need to decompose it each step or use Cholesky.
        # For small n (5), Cholesky or SVD is cheap.
        
        eval_count = 0
        best_x = mean.copy()
        best_f = float('inf')
        
        # Initial evaluation of the mean
        f_mean = func(mean)
        eval_count += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = mean.copy()
            
        # Main loop
        while eval_count < total_budget:
            # Determine how many points we can evaluate in this generation
            # We must evaluate 'lam' points. If we don't have enough budget, break.
            if eval_count + lam > total_budget:
                # We can still evaluate the remaining points, but CMA-ES update requires the full set.
                # If we can't complete a generation, we might just do random search or stop.
                # To maximize AOCC, we should keep making progress.
                # If we can't do a full CMA step, we can just sample the remaining budget from the current distribution.
                remaining = total_budget - eval_count
                for _ in range(remaining):
                    # Sample from current distribution
                    z = rng.normal(size=n)
                    # Transform to x
                    # Need Cholesky of C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # Fallback if C is not positive definite
                        L = np.eye(n)
                    x = mean + sigma * (L @ z)
                    x = np.clip(x, -5, 5)
                    f = func(x)
                    eval_count += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                break
            
            # Sample population
            # Get Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or fix it
                C = np.eye(n)
                L = np.eye(n)
            
            # Generate random normal vectors
            Z = rng.normal(size=(lam, n))
            # Transform to x
            # X = mean + sigma * (Z @ L.T)  # If L is lower triangular from Cholesky
            X = mean + sigma * (Z @ L.T)
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                F[i] = func(X[i])
                eval_count += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
            
            # Sort by function value
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Update mean
            # Mean = sum(weights_i * X_sorted[i])
            mean_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update step size (sigma)
            # p_sigma = (1 - c_s) * p_sigma + sqrt(c_s * (2 - c_s) * mueff) / sigma * (C^{-1/2} * (mean_new - mean))
            # We need C^{-1/2}. We can use the inverse of the Cholesky factor or SVD.
            # For small n, let's use SVD to get C^{-1/2}
            U, s, Vt = np.linalg.svd(C)
            # C^{-1/2} = V * (1/sqrt(s)) * U^T
            C_inv_sqrt = Vt.T @ np.diag(1.0 / np.sqrt(s + 1e-16)) @ U.T
            
            delta_mean = mean_new - mean
            norm_delta = np.linalg.norm(C_inv_sqrt @ delta_mean)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2)) # Expected norm of standard normal in n-dim
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ delta_mean) / (sigma * chi_n)
            
            # Hessian potential for step size adaptation
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*(eval_count/lam+1))) ) < (1.4 + 2/(n+1)) * chi_n
            hsig = 1.0 if hsig else 0.0
            
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            # p_c = (1 - c_c) * p_c + hsig * sqrt(c_c * (2 - c_c) * mueff) * (delta_mean / sigma)
            cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (delta_mean / sigma)
            
            # C = (1 - mu_1 - mu_mu + c1) * C + c1 * (pc @ pc.T + (1 - hsig) * cc * (2 - cc) * I) + cmu * sum(weights_i * (x_i - mean_old) @ (x_i - mean_old).T)
            # Note: The update uses the difference from the *old* mean
            # Let's compute the rank-1 and rank-mu updates
            # Rank-1 update term
            # The term (1 - mu_1 - mu_mu + c1) is the decay factor for C
            # mu_1 = c1 * (1 - c1)
            # mu_mu = cmu
            
            # Actually, the standard update is:
            # C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - hsig) * cc * (2 - cc) * I) + cmu * sum(weights_i * (y_i @ y_i.T))
            # where y_i = (x_i - mean_old) / sigma
            
            y = (X_sorted[:mu] - mean) / sigma
            rank_mu_update = np.sum(weights[:, None, None] * (y[:, :, None] * y[:, None, :]), axis=0)
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * np.eye(n)) + cmu * rank_mu_update
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            C += 1e-16 * np.eye(n)
            
            # Update mean
            mean = mean_new
            
        return best_x, best_f
# EVOLVE-BLOCK-END
