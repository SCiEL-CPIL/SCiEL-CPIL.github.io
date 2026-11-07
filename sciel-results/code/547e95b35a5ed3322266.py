"""Rewritten optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """CMA-ES optimizer for black-box continuous functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lambda_pop = 4 + int(3 * np.log(n))
        
        # Weights
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / sum(weights**2)
        
        # Step size
        sigma = 1.0
        
        # Initial mean
        x_mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Eigenvalues and eigenvectors of C (updated periodically)
        C_eigvals = np.ones(n)
        C_eigvecs = np.eye(n)
        
        # Adaptation paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        
        # Damping for step size
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        
        # Threshold for updating C
        hsig = (1.4 + 2 / (n + 1)) * (1 - np.prod(np.exp(-2 * c_sigma * (1 - np.exp(-c_sigma)))))
        
        calls = 0
        best_x = x_mean.copy()
        best_fx = np.inf
        
        # Initial evaluation
        fx = func(x_mean)
        calls += 1
        if fx < best_fx:
            best_fx = fx
            best_x = x_mean.copy()
            
        # Main loop
        while calls < budget:
            # Generate population
            Y = np.zeros((lambda_pop, n))
            for i in range(lambda_pop):
                # Sample from N(0, C) using eigendecomposition
                z = np.random.standard_normal(n)
                Y[i] = np.dot(C_eigvecs, np.sqrt(C_eigvals) * z)
            
            # Evaluate population
            X = x_mean + sigma * Y
            F = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                # Clip to box [-5, 5]
                X[i] = np.clip(X[i], -5, 5)
                F[i] = func(X[i])
                calls += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = X[i].copy()
                    
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            x_old = x_mean.copy()
            x_mean = sum(weights[i] * X_sorted[i] for i in range(mu))
            
            # Update adaptation paths
            # p_sigma
            C_sqrt_inv = np.dot(np.dot(C_eigvecs, np.diag(1.0 / np.sqrt(C_eigvals))), C_eigvecs.T)
            y_sigma = np.dot(C_sqrt_inv, (x_mean - x_old) / sigma)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            
            # hsig
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls // lambda_pop + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
            
            # p_c
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y_sigma
            
            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_c * (2 - c_c) - c_sigma) * C + \
                c_sigma * (np.outer(p_c, p_c) + delta_hsig * C) + \
                c_sigma / mueff * sum(weights[i] * np.outer(X_sorted[i] - x_old, X_sorted[i] - x_old) for i in range(mu))
            
            # Ensure positive semi-definiteness and fix numerical issues
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_eigvals = eigvals
            C_eigvecs = eigvecs
            
            # Update step size
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Bounds handling: if mean goes out of bounds, pull it back
            # Simple strategy: clip mean
            x_mean = np.clip(x_mean, -5, 5)
            
            # Convergence check
            if np.linalg.norm(x_mean - x_old) < 1e-8 * max(1, np.linalg.norm(x_mean)) and sigma < 1e-6:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
