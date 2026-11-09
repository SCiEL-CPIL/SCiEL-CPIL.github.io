"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # CMA-ES parameters
        d = self.dim
        # Population size lambda
        # For small d, lambda = 4 + floor(3 * ln(mu)) is standard, but fixed 10 is often robust for d=5
        # Let's use lambda = 10 for d=5. If d was larger, we'd adjust.
        lam = 10
        mu = lam // 2  # number of parents
        
        # Recombination weights
        # Standard weights: w_i = log(mu + 0.5) - log(i)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Step size adaptation parameters
        cs = (mueff + 2) / (lam + 4)
        # Covariance matrix adaptation parameters
        cc = (4 + mueff / lam) / (lam + 4 + 2 * mueff / lam)
        c1 = 2 / ((lam + 1.3)**2 + mueff)
        cmu = 2 * (mueff - 2 + 1.0 / mueff) / ((lam + 2)**2 + mueff)
        
        # Initialization
        # Start at the center of the box, which is 0. 
        # The problem states minimizer is in [-4, 4]^5. 0 is a good guess.
        # However, to be robust against potential large shifts (though bounded), 
        # random start is safer. But CMA-ES converges fast. 
        # Let's start at 0.
        x_mean = np.zeros(d)
        sigma = 2.0 # Initial step size. Box is [-5, 5], so 2 is reasonable.
        
        # Covariance matrix
        C = np.eye(d)
        # Square root of C
        C_sqrt = np.eye(d)
        # p_sigma and p_c
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        # Pre-generate normal matrix for efficiency? No, rng is fine.
        
        n_gen = self.budget // lam
        if n_gen < 1:
            n_gen = 1
            
        # We have budget evaluations.
        # Each generation uses lam evaluations.
        # We can run n_gen generations.
        
        for gen in range(n_gen):
            if time.time() > deadline:
                break
                
            # Generate population
            Y = rng.standard_normal((lam, d))
            # Transform to x
            X = np.zeros((lam, d))
            for i in range(lam):
                X[i] = x_mean + sigma * (Y[i] @ C_sqrt.T)
                
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                # Clip to box
                x_i = np.clip(X[i], -5, 5)
                F[i] = func(x_i)
                
                # Track global best
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_i.copy()
                    
            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update mean
            # x_new = sum(w_i * x_{i:lambda})
            x_new = np.zeros(d)
            for i in range(mu):
                x_new += w[i] * X[i]
                
            # Update step size
            # p_sigma = (1 - cs) * p_sigma + sqrt(cs * (2 - cs) * mueff) * C_sqrt^{-1} * (x_new - x_mean) / sigma
            # Note: C_sqrt^{-1} is needed. We can maintain C_sqrt and its inverse or use Cholesky.
            # For simplicity and speed in d=5, we can just compute the inverse of C_sqrt.
            # Actually, standard CMA-ES updates p_sigma using the "natural gradient" direction.
            # direction = C_sqrt^{-1} @ (x_new - x_mean) / sigma
            # Inverse of C_sqrt can be computed via solve.
            
            # Efficient update:
            # We need C_sqrt_inv.
            # Since d is small (5), we can just invert.
            try:
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_sqrt_inv = np.eye(d)
                
            diff = (x_new - x_mean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt_inv @ diff)
            
            # h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            norm_p_sigma_target = np.sqrt(d) * (1 - (1 - cs)**(1 / (self.budget / lam))) # Approximate
            # More accurate: 1 - (1 - cs)^(1/(budget/lam)) is for infinite budget.
            # Let's use the standard approximation: sqrt(d) * (1 - (1-cs)^(1/(lam))) is not right.
            # Standard: h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-cs)^2t) < (1.4 + 2/(d+1)) * E(N(0,1))
            # E(N(0,1)) = sqrt(d)
            # Let's use a simpler heuristic or standard formula.
            # h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 2/(d+1)) * np.sqrt(d) else 0.0
            # This is a common simplification.
            
            # Update p_c
            # p_c = (1 - cc) * p_c + h_sigma * sqrt(cc * (2 - cc) * mueff) * (x_new - x_mean) / sigma
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 2/(d+1)) * np.sqrt(d) else 0.0
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * diff
            
            # Update sigma
            # sigma = sigma * exp(cs / dsigma * (||p_sigma|| / E||N(0, I)|| - 1))
            # dsigma = 1 + 2 * max(0, (mueff - 1)/(d + 1))
            dsigma = 1 + 2 * max(0, (mueff - 1) / (d + 1))
            sigma = sigma * np.exp(cs / dsigma * (norm_p_sigma / np.sqrt(d) - 1))
            
            # Update C
            # C = (1 - c1 - cmu) * C + c1 * (p_c * p_c + (1 - h_sigma) * cc * (2 - cc) * C) + cmu * sum(w_i * (x_i - x_mean)(x_i - x_mean)^T)
            
            # Term 1: (1 - c1 - cmu) * C
            C_new = (1 - c1 - cmu) * C
            
            # Term 2: c1 * (p_c * p_c + (1 - h_sigma) * cc * (2 - cc) * C)
            C_new += c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * cc * (2 - cc) * C)
            
            # Term 3: cmu * sum(w_i * (x_i - x_mean)(x_i - x_mean)^T)
            # We use the sorted X (top mu)
            for i in range(mu):
                diff_i = (X[i] - x_mean) / sigma # Standard CMA-ES uses normalized difference? 
                # Actually, the standard formula is:
                # C += cmu * sum(w_i * (y_i * y_i^T) where y_i = C_sqrt^{-1}(x_i - x_old)/sigma
                # Let's stick to the unnormalized version if we adjust weights, 
                # but standard CMA-ES uses the normalized directions in the covariance update?
                # No, the standard update is:
                # C = (1 - c1 - cmu)C + c1(p_c p_c^T + ...) + cmu sum w_i (x_i - x_old)(x_i - x_old)^T / sigma^2
                # Wait, the covariance matrix C represents the shape of the distribution in the "y" space (standard normal).
                # The x-space covariance is sigma^2 C.
                # The update rule for C is:
                # C <- (1 - c1 - cmu) C + c1 (p_c p_c^T + (1-h) cc(2-cc) C) + cmu sum w_i y_i y_i^T
                # where y_i = C_sqrt^{-1} (x_i - x_old) / sigma
                
                y_i = C_sqrt_inv @ ((X[i] - x_mean) / sigma)
                C_new += cmu * w[i] * np.outer(y_i, y_i)
                
            C = C_new
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Eigenvalue check for numerical stability
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.sqrt(np.diag(eigvals))
            except np.linalg.LinAlgError:
                # Fallback to identity if it fails
                C = np.eye(d)
                C_sqrt = np.eye(d)
                
            x_mean = x_new

        return best_x, best_f

# EVOLVE-BLOCK-END
