import numpy as np
import time
from scipy.linalg import fractional_matrix_power

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        
        # IPOP-CMA-ES parameters
        lam = 7 
        max_lam = 20
        
        def get_weights(lam):
            mu = lam // 2
            weights = np.zeros(lam)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= np.sum(weights)
            return mu, weights

        # State
        rng = np.random.default_rng(0)
        x_mean = rng.uniform(-4, 4, n)
        sigma = 1.0
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        mu, weights = get_weights(lam)
        mueff = 1.0 / np.sum(weights[:mu]**2)
        
        # Adaptation parameters
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 2) / (n + 2)) - 1) + cs
        cc = (4 - mueff) / (n + 4)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        
        x_best = x_mean.copy()
        f_best = func(x_best)
        calls = 1
        stagnation_count = 0
        
        restart_threshold = 1e-8
        max_stagnation = 30
        
        while calls < budget:
            if time.time() - self.start_time > 290:
                break
            
            # Generate population
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)
            
            z = rng.standard_normal((lam, n))
            x_pop = x_mean + sigma * (z @ B.T)
            x_pop = np.clip(x_pop, -5, 5)
            
            f_pop = np.array([func(x) for x in x_pop])
            calls += lam
            
            idx = np.argsort(f_pop)
            x_sorted = x_pop[idx]
            f_sorted = f_pop[idx]
            
            # Update best
            if f_sorted[0] < f_best:
                f_best = f_sorted[0]
                x_best = x_sorted[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            # Update mean
            x_new_mean = np.sum(weights[:mu, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Update paths
            try:
                Cinv_sqrt = fractional_matrix_power(C, -0.5)
            except:
                Cinv_sqrt = np.eye(n)
                
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cinv_sqrt @ (x_new_mean - x_mean) / sigma
            norm_ps = np.linalg.norm(ps)
            threshold = (1.4 + 2/(n+1)) * np.sqrt(1 - (1-cs)**2)
            hsig = 1 if norm_ps / np.sqrt(1 - (1-cs)**2) <= threshold else 0
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_new_mean - x_mean) / sigma
            
            # Update C
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * (delta_hsig - 1) + cmu) * C
            C += c1 * delta_hsig * (pc @ pc.T)
            for i in range(mu):
                diff = (x_sorted[i] - x_mean) / sigma
                C += cmu * weights[i] * (diff @ diff.T)
                
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(2*n) - 1))
            
            # Update mean
            x_mean = x_new_mean
            
            # Check for restart
            spread = np.linalg.norm(x_sorted[0] - x_sorted[-1])
            if spread < restart_threshold or sigma < restart_threshold or stagnation_count > max_stagnation:
                lam = min(2 * lam, max_lam)
                mu, weights = get_weights(lam)
                mueff = 1.0 / np.sum(weights[:mu]**2)
                cs = (mueff + 2) / (n + mueff + 5)
                ds = 1 + 2 * max(0, np.sqrt((mueff - 2) / (n + 2)) - 1) + cs
                cc = (4 - mueff) / (n + 4)
                c1 = 2 / ((n + 1.3)**2 + mueff)
                cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
                
                x_mean = x_best.copy()
                sigma = max(sigma * 2, 0.5)
                C = np.eye(n)
                ps = np.zeros(n)
                pc = np.zeros(n)
                stagnation_count = 0
                
        return x_best, f_best
