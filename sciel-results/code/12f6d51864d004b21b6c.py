import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        n = self.dim
        budget = self.budget
        
        lambda_pop = 4 + int(3 * np.log(n))
        if lambda_pop < 10: lambda_pop = 10
        mu = lambda_pop // 2
        
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        cs = (mueff + 2) / (n + mueff + 5)
        cc = (4 + mueff / lambda_pop) / (n + 4 + mueff / lambda_pop)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        dsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        
        x_mean = np.random.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        hs = np.sqrt((1 + cc) * n + 2)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        evals = 1
        
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Sample z ~ N(0, I)
            z = np.random.randn(lambda_pop, n)
            
            # Transform to y
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset C to I
                C = np.eye(n)
                C_chol = np.eye(n)
                sigma *= 0.5
                
            y = z @ C_chol.T
            x_pop = x_mean[None, :] + sigma * y
            x_pop = np.clip(x_pop, -5, 5)
            
            f_pop = np.array([func(x) for x in x_pop])
            evals += lambda_pop
            
            idx = np.argsort(f_pop)
            x_sorted = x_pop[idx]
            f_sorted = f_pop[idx]
            y_sorted = y[idx]
            
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
                
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            y_w = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_w / sigma
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**((evals + 1) / dsigma)) <= (1.4 + 2 / (n + 1)) * hs)
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w / sigma
            
            delta_hs = 1 - hsig
            C = (1 + c1 * delta_hs + cmu - c1) * C \
                + c1 * (np.outer(pc, pc) - (1 - (1 - cc)**2) * C) \
                + cmu * np.sum(weights[:mu, np.newaxis, np.newaxis] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0)
            
            C = (C + C.T) / 2
            
            sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / hs - 1))
            
            if sigma < 1e-8:
                break

        return best_x, best_f
