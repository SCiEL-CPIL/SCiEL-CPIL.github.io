"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard: 330s limit, leave safety margin
        start_time = time.time()
        deadline = start_time + 320.0
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        if lam < 10: lam = 10
        
        # Initial point and sigma
        x = np.zeros(dim)
        sigma = 2.0
        
        # Covariance matrix C (initially identity)
        C = np.eye(dim)
        
        # Evolution path for sigma (cs) and for covariance (C)
        cs = 1. / (np.sqrt(dim) * (0.5 + 1. / (4. * dim) + 2. / (dim * lam)))
        ds = 1. + 2. * max(0., np.sqrt((lam - 2.) / (dim + 2.)) - 1.) + cs
        mu = lam // 2
        weights = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1. / np.sum(weights**2)
        c1 = (2. * (mueff + 2.) / ((dim + 2.)**2 + mueff)) / ds
        cmu = min(1. - c1, 2. * (mueff - 2. + 1. / mueff) / ((dim + 2.)**2 + mueff))
        c_sigma = (mueff + 2.) / (dim + mueff + 5.)
        # d_sigma correction (simplified)
        d_sigma = 1. + 2. * max(0., np.sqrt((mueff - 1.) / (dim + 1.)) - 1.) + c_sigma
        
        # Cholesky factor of C
        C_sqrt = np.linalg.cholesky(C)
        
        # Initial evaluation
        x = np.clip(x, -5, 5)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        evals = 1
        
        # Paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        while evals < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate population
            Y = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.randn(dim)
                Y[i] = x + sigma * (C_sqrt @ z)
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                if evals >= budget or time.time() > deadline:
                    # If we run out of budget/time, fill remaining with worst possible value to stop loop
                    F[i] = np.inf
                    continue
                y_i = np.clip(Y[i], -5, 5)
                F[i] = func(y_i)
                evals += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = y_i.copy()
            
            # Sort indices by fitness
            order = np.argsort(F)
            
            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * Y[order[i]]
            
            # Update sigma
            # p_sigma = (1-cs)*p_sigma + cs/sqrt(mueff) * C_sqrt^{-1} * (x_new - x)/sigma
            # Note: C_sqrt^{-1} is inverse of Cholesky factor
            C_sqrt_inv = np.linalg.inv(C_sqrt)
            p_sigma = (1 - cs) * p_sigma + cs / np.sqrt(mueff) * (C_sqrt_inv @ (x_new - x) / sigma)
            
            # Hessian norm approximation for chi-normal correction
            chi_norm = np.sqrt(dim)
            # Simplified chi_norm correction, often approximated or omitted in simple implementations
            # For robustness, we use the standard update
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_norm - 1.))
            
            # Update covariance
            # p_c = (1-c1)*p_c + c1/sqrt(mueff) * (x_new - x)/sigma
            p_c = (1 - c1) * p_c + c1 / np.sqrt(mueff) * (x_new - x) / sigma
            
            # Update C
            # C = (1 - c1 - cmu + c1*mu*delta(p_c^2 - (1-2*c1))*C) * C 
            #     + cmu * sum(weights[i] * (y_i - x)(y_i - x)^T)
            
            # Term 1: Rank-one update from p_c
            pc_sq = np.dot(p_c, p_c)
            delta_corr = c1 * (mu - 2 + 1 / mu) * (pc_sq / sigma**2 - (1 - 2 * c1))
            
            # Term 2: Rank-mu update from top mu individuals
            H = np.zeros((dim, dim))
            for i in range(mu):
                diff = Y[order[i]] - x
                H += weights[i] * np.outer(diff, diff)
            
            C = (1 - c1 - cmu + c1 * mu * delta_corr) * C + cmu * H / sigma**2
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2.0
            
            # Update Cholesky factor
            # Check if C is positive definite
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not PD, add small regularization
                C += 1e-12 * np.eye(dim)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset C to identity if badly broken
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
            
            x = x_new
        
        return best_x, best_f

# EVOLVE-BLOCK-END
