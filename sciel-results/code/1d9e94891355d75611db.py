import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lam = 10
        w = np.log(lam/2 + 1) - np.log(np.arange(1, lam+1))
        w = w / np.sum(w)
        mu = int(lam / 2)
        mueff = int(np.sum(w[:mu])**2 / np.sum(w**2))
        
        # Initial state
        x_m = np.zeros(dim)
        f_m = func(x_m)
        calls = 1
        best_x = x_m.copy()
        best_f = f_m
        
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(c_1 / 2, (mueff - 2) / (mueff + 2))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        gen = 0
        
        while calls < budget:
            if time.time() > deadline - 5.0:
                break
            
            gen += 1
            # Generate samples
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            
            samples = []
            for i in range(lam):
                y = rng.standard_normal(dim)
                x_i = x_m + sigma * (L @ y)
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                samples.append((f_i, x_i))
                
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            samples.sort(key=lambda t: t[0])
            
            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * samples[i][1]
            
            # Update step size sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L @ (x_new - x_m) / sigma)
            
            norm_ps = np.linalg.norm(ps)
            threshold = (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma)**(2 * gen))
            
            if norm_ps / chiN < threshold:
                sigma *= np.exp((1/5) * (norm_ps/chiN - 1))
                h_sigma = 1
            else:
                sigma *= np.exp((1/5) * (norm_ps/chiN - 1)) # Wait, sigma update is always based on ps norm
                h_sigma = 0
            # Actually, sigma update is: sigma = sigma * exp(1/5 * (||ps||/chiN - 1))
            # This is always applied. h_sigma is for the covariance update.
            
            # Update pc
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x_m) / sigma
            
            # Update covariance matrix C
            # z_i = (x_i - x_m) / sigma
            # C = (1 - c1 - cmu + c1 * h_sigma * c_c * (2 - c_c)) * C 
            #     + c1 * outer(pc, pc) 
            #     + cmu * sum(w_i * outer(z_i, z_i))
            
            factor = 1 - c_1 - c_mu + c_1 * h_sigma * c_c * (2 - c_c)
            C = factor * C + c_1 * np.outer(pc, pc)
            
            for i in range(mu):
                z_i = (samples[i][1] - x_m) / sigma
                C += c_mu * w[i] * np.outer(z_i, z_i)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            x_m = x_new
            
        return best_x, best_f
