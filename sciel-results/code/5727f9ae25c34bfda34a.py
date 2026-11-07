import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285

        # CMA-ES Parameters
        # For n=5, lambda=10 is a good choice
        lam = 10
        # Weights
        w = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
        w /= np.sum(w)
        mu = int(lam / 2)  # Number of parents
        w = w[:mu]
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Step size
        sigma = 1.0
        # Mean
        m = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Eigen decomposition cache
        eigvals = np.ones(n)
        eigvecs = np.eye(n)
        
        # Counters
        calls = 0
        best_x = m.copy()
        best_f = func(m)
        calls += 1
        
        # Helper to update eigen decomposition
        def update_eigen(C):
            nonlocal eigvals, eigvecs, C_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            # Update C_sqrt: C_sqrt = eigvecs * sqrt(eigvals)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            
        update_eigen(C)
        
        # Cumulative step size control state
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Main Loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if time.time() > deadline or calls >= budget:
                    break
                z = np.random.standard_normal(n)
                y = C_sqrt @ z
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                f = func(x)
                calls += 1
                ys.append(y)
                xs.append(x)
                fs.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(xs) < lam:
                break
            
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # Sort by function value
            order = np.argsort(fs)
            ys = ys[order]
            xs = xs[order]
            fs = fs[order]
            
            # Update Mean
            m_old = m
            m = np.sum(w[:, np.newaxis] * xs[:mu], axis=0)
            
            # Update Step Size
            y_mean = (m - m_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / np.linalg.norm(C_sqrt) * C_sqrt @ y_mean
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
            
            # Update Covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_mean
            h_sig = 1 if np.linalg.norm(p_sigma / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2 / (n + 1)) * np.sqrt(mueff) else 0
            rank_mu = np.sum(w[:, np.newaxis] * (ys[:mu] * ys[:mu].T), axis=0)
            rank_one = np.outer(p_c, p_c)
            C = (1 + c_1 * h_sig * (n + 1) - c_1 - c_mu) * C + c_mu * rank_mu + c_1 * h_sig * (rank_one - C)
            
            # Ensure positive definite and update eigen
            # Add small regularization if needed
            C = (C + C.T) / 2
            try:
                update_eigen(C)
            except:
                # Fallback if eigen fails
                C = np.eye(n)
                update_eigen(C)
                
        return best_x, best_f
