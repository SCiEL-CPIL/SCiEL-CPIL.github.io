"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style Schwefel function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / (dim**1.7 + 10)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim**2 + 18))
        
        # Initial parameters
        # Start at origin, which is a good heuristic for BBOB functions
        # Initial step size 2.0 is standard for [-5,5] box
        x_m = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = x_m.copy()
        best_f = np.inf
        calls_made = 0
        
        # Main loop
        while calls_made + lam <= budget:
            if time.time() > deadline:
                break
            
            # Generate population
            Y = []
            X = []
            for i in range(lam):
                # Sample from N(0, C)
                # Cholesky decomposition for sampling
                L = np.linalg.cholesky(C)
                z = np.random.normal(0, 1, dim)
                y = L @ z
                x = x_m + sigma * y
                Y.append(y)
                X.append(x)
            
            # Evaluate
            F = []
            for x in X:
                fx = func(x)
                calls_made += 1
                F.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # Sort
            idx = np.argsort(F)
            X = [X[i] for i in idx]
            Y = [Y[i] for i in idx]
            F = [F[i] for i in idx]
            
            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * X[i]
            x_m = x_new
            
            # Update step size (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ (x_new - x_m))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(dim+1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps)/chi_n - 1))
            
            # Update covariance matrix
            # Rank-one update
            pc = (1 - c_c) * pc + c_c * (x_new - x_m) / sigma
            delta_sig = 1 - np.exp(-c_c * (np.linalg.norm(pc)**2 / dim - 1))
            C = (1 + c_1 * c_mu + c_1 * delta_sig - c_1 - c_mu) * C
            
            # Rank-mu update
            for i in range(mu):
                y_i = (X[i] - x_m) / sigma
                C += c_mu * np.outer(y_i, y_i)
                
            # Add rank-one part
            C += c_1 * np.outer(pc, pc)
            
            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            # If C becomes singular or non-PD, reset
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                sigma = 2.0
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                x_m = np.zeros(dim)
                break # Restart if it breaks

        return best_x, best_f
# EVOLVE-BLOCK-END
