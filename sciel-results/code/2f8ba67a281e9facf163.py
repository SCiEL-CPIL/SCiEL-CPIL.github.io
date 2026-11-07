"""CMA-ES based optimizer for BBOB f17."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    CMA-ES is a state-of-the-art derivative-free optimizer that adapts the
    covariance matrix of a multivariate normal distribution to the local
    landscape structure. This is particularly effective for rotated/shifted
    landscapes like BBOB functions where axis-aligned methods struggle.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, use 300s to be safe
        deadline = start_time + 300
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        sigma = 2.0
        
        # Population size
        lambda_ = 12 
        
        # Initial mean
        mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Step sizes for parameters
        cmu = 1.0 / (4 + np.exp(n / 20))
        ccs = (4 + n / lambda_) / (n + 4 + 2 * n / lambda_)
        dsigma = 1 + 2 * max(0, np.sqrt((lambda_ / 2 - 1) / (n + 1)) - 1) + ccs
        csigma = (1 + ccs) / (n + dsigma)
        cc = (2 + 2 * np.sqrt((lambda_ / 2 - 1) / (n + 1))) / (n + 2)
        
        # State variables
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Path variables for cumulative step size adaptation
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Helper to clip to box
        def clip(x):
            return np.clip(x, -5, 5)
        
        # Evaluate function wrapper
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_clipped = clip(x)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx
        
        # Initial evaluation of mean
        f_mean = evaluate(mean)
        
        # Main loop
        while calls < budget and time.time() < deadline:
            # Generate population
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
            
            # Sample lambda_ points
            Z = np.random.randn(lambda_, n)
            X = mean + sigma * (Z @ L.T)
            
            # Clip population to box
            X = np.clip(X, -5, 5)
            
            # Evaluate population
            F = np.array([evaluate(x) for x in X])
            
            # Check if we ran out of budget
            if calls >= budget:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Top mu individuals
            mu = max(1, int(lambda_ / 2))
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            
            # mu_eff
            mu_eff = 1.0 / np.sum(w**2)
            
            # Update mean
            if np.isinf(F_sorted[0]):
                break
            
            X_top = X_sorted[:mu]
            mean_new = np.sum(w[:, np.newaxis] * X_top, axis=0)
            
            # Update path p_sigma
            v = (mean_new - mean) / sigma
            try:
                y = np.linalg.solve(L, v)
            except np.linalg.LinAlgError:
                y = v
            
            p_sigma = (1 - csigma) * p_sigma + csigma * np.sqrt(mu_eff) * y
            
            # Check step size adaptation
            E_norm = np.sqrt(n) * (1 - 1.0/(4.0*n))
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(csigma / dsigma * (norm_p_sigma / E_norm - 1))
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update path p_c
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                C_inv_half_v = C_inv_half @ v
            except np.linalg.LinAlgError:
                C_inv_half_v = v
            
            p_c = (1 - cc) * p_c + cc * np.sqrt(mu_eff) * C_inv_half_v
            
            # Update Covariance Matrix C
            Y = (X_top - mean) / sigma
            try:
                Y_std = np.linalg.solve(L, Y.T).T
            except np.linalg.LinAlgError:
                Y_std = Y
            
            c_mu_base = 2.0 * (mu_eff - 2.0 + 1.0/mu_eff) / ((n + 2.0)**2 + mu_eff)
            c_mu = c_mu_base * (1.0 - csigma / dsigma * (norm_p_sigma**2 - 1.0))
            c_mu = np.clip(c_mu, 0.0, 2.0 / mu_eff)
            
            outer_sum = Y_std.T @ (w[:, np.newaxis] * Y_std)
            
            C = (1 - c_mu) * C + c_mu * outer_sum
            C = (C + C.T) / 2.0
            
            # Update mean
            mean = mean_new
            
            # Check for convergence or stagnation
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
