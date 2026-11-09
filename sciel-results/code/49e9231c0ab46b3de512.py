import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        calls = [0]
        best_x = None
        best_f = float('inf')
        
        def wrapped_func(x):
            nonlocal best_x, best_f
            if calls[0] >= self.budget or time.time() > deadline:
                return float('inf')
            calls[0] += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        dim = self.dim
        budget = self.budget
        
        # Phase 1: Random sampling for initial exploration
        n_init = 10
        if budget > n_init:
            for _ in range(n_init):
                if calls[0] >= budget or time.time() > deadline:
                    break
                x = np.random.uniform(-5, 5, dim)
                wrapped_func(x)
        
        if best_x is None:
            return best_x, best_f
        
        if calls[0] >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: CMA-ES-like optimization
        # Parameters
        lam = 12  # Population size
        mueff = 6  # Effective population size for adaptation
        
        # Initial sigma
        sigma = 0.5
        
        # State
        x_mean = best_x.copy()
        C = np.eye(dim)
        C_inv = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # CMA-ES constants
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Generate initial population
        def generate_population(x_mean, sigma, C, B, D):
            z = np.random.randn(lam, dim)
            y = (z @ B * D).T @ np.eye(dim)  # This is wrong, need proper transformation
            # Correct transformation: y = B * (z * D)
            # Actually, y_i = B @ (z_i * D)
            y = np.zeros((lam, dim))
            for i in range(lam):
                y[i] = B @ (z[i] * D)
            x = x_mean + sigma * y
            return x, y
        
        # CMA-ES loop
        while calls[0] < budget and time.time() < deadline:
            # Generate population
            pop_x, pop_y = generate_population(x_mean, sigma, C, B, D)
            
            # Evaluate population
            pop_f = np.zeros(lam)
            for i in range(lam):
                if calls[0] >= budget or time.time() > deadline:
                    break
                pop_f[i] = wrapped_func(pop_x[i])
            
            # Check if we ran out of budget
            if calls[0] >= budget or time.time() > deadline:
                break
            
            # Sort by fitness
            idx = np.argsort(pop_f)
            pop_x = pop_x[idx]
            pop_y = pop_y[idx]
            pop_f = pop_f[idx]
            
            # Update mean
            x_new = np.sum(weights[:, None] * pop_x[:mu], axis=0)
            
            # Update covariance matrix
            y_w = (pop_y[:mu] - x_mean / sigma).T
            C_new = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (delta_sigma * c_sigma * (1 - c_sigma)) * C) + c_mu * np.sum(weights[:, None] * np.outer(y_w.T, y_w.T), axis=0)
            
            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) / chi_n * np.linalg.norm(y_w.T @ (B.T * D))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update evolution path
            p_c = (1 - c_c) * p_c + c_c / np.sqrt(c_c * (2 - c_c)) * (np.sqrt(mueff) / np.linalg.norm(C @ np.eye(dim))) * (x_new - x_mean) / sigma
            
            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C_new)
            eigvals = np.clip(eigvals, 1e-12, 1e12)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Check for stagnation
            if np.linalg.norm(p_c) > np.sqrt((2 + 4 / dim) / (1 - c_c)) * chi_n:
                # Restart
                x_mean = x_new
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_sigma = np.zeros(dim)
                sigma = 0.5
            else:
                x_mean = x_new
        
        return best_x, best_f
