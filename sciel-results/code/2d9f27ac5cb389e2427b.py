import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 290.0  # Leave 10s margin before 300s kill

        def time_left():
            return deadline - time.time()

        # CMA-ES implementation for multimodal, rotated/shifted landscapes
        # This adapts the search distribution to the problem structure
        
        # Initial population
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~9 for dim=5
        lambda_ = pop_size
        
        # Initial mean
        mean = np.zeros(dim)
        
        # Initial sigma
        sigma = 1.0
        
        # CMA-ES parameters
        alpha_mu = 0.5
        mu = lambda_ // 2  # number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (alpha_mu * mu_eff - 2) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        # Eigen decomposition for step size
        eigvals, eigvecs = np.linalg.eigh(C)
        D = np.sqrt(eigvals)
        inv_D = 1.0 / np.maximum(D, 1e-10)
        
        # Function call tracking
        calls = 0
        best_x = mean.copy()
        best_f = np.inf
        
        # Evaluate initial mean
        best_f = func(best_x)
        calls += 1
        
        # Main loop
        while calls < budget and time_left() > 2.0:
            # Generate population
            Y = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                Y[i] = sigma * np.dot(eigvecs, (D * z))
            
            # Evaluate population
            F = np.zeros(lambda_)
            for i in range(lambda_):
                x_i = mean + Y[i]
                F[i] = func(x_i)
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_i.copy()
            
            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Update mean
            mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0) + mean
            
            # Update path variables
            z_sigma = np.dot(eigvecs.T, (D * np.dot(eigvecs.T, mean - (mean - Y_sorted[0, :]))))
            # Simplified: z_sigma = sigma * dot(eigvecs.T, D * eigvecs.T * (mean_new - mean_old))
            # Let's use a simpler formulation
            z_sigma = np.zeros(dim)
            for i in range(dim):
                z_sigma[i] = np.sum(Y_sorted[0, :] * eigvecs[i, :]) / D[i] if D[i] > 1e-10 else 0
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * np.dot(eigvecs, (D * z_sigma))
            
            # Update covariance matrix
            h_sigma = (np.sqrt((1 - (1 - c_sigma)**2 * 10)) * np.linalg.norm(p_sigma) / np.sqrt(np.trace(C)) - (1 - c_sigma)) / np.sqrt(1 - (1 - c_sigma)**2 * 10)
            
            delta_h = 1 - h_sigma if h_sigma > 0.9 else 0
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu], Y_sorted[:mu]), axis=0)
            
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
            
            # Update sigma
            sigma *= np.exp(c_sigma / (2 * c_c) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            D = np.sqrt(eigvals)
            
            # Clamp sigma
            sigma = max(1e-8, min(sigma, 10.0))
            
            # Check for convergence
            if np.max(D) / np.min(D) > 1e8:
                break
        
        return best_x, best_f
