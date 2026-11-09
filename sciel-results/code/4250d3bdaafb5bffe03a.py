import math
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Initial guess: center of the box
        start_mean = np.zeros(self.dim)
        start_sigma = 1.0
        
        best_x, best_f, calls_used = self._run_cmaes(func, start_mean, start_sigma, self.budget, rng, deadline)
        
        # Optional: Polish with L-BFGS-B if time and budget remain
        # The directive suggested L-BFGS-B. Let's try a quick polish.
        if time.time() < deadline and calls_used < self.budget:
            try:
                # Use the best found so far as start point for L-BFGS-B
                # L-BFGS-B might help if we are close to a local min
                res = minimize(func, best_x, method='L-BFGS-B', bounds=[(-5, 5)] * self.dim, 
                               options={'maxiter': 100, 'maxfun': self.budget - calls_used})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except:
                pass

        return best_x

    def _run_cmaes(self, func, start_mean, start_sigma, budget, rng, deadline):
        dim = self.dim
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        mean = start_mean.copy()
        sigma = start_sigma
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        calls_used = 0
        best_x = mean.copy()
        best_f = func(best_x)
        calls_used += 1

        while calls_used < budget:
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lam, dim))
            try:
                L = np.linalg.cholesky(C)
            except:
                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C)
                except:
                    break

            for i in range(lam):
                z = rng.standard_normal(dim)
                pop[i] = mean + sigma * (L @ z)

            # Evaluate population
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls_used >= budget or time.time() > deadline:
                    break
                fvals[i] = func(pop[i])
                calls_used += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            if calls_used >= budget or time.time() > deadline:
                break

            # Sort
            order = np.argsort(fvals)
            y = np.zeros((mu, dim))
            for i in range(mu):
                y[i] = pop[order[i]]

            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * y, axis=0)

            # Update paths
            v = (new_mean - mean) / sigma
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            ps = (1 - c_sigma) * ps + c_sigma / math.sqrt(c_sigma) * C_inv_sqrt @ v
            
            norm_ps = np.linalg.norm(ps)
            t_gen = calls_used / lam
            h_sigma = (norm_ps / math.sqrt(1 - (1 - c_sigma)**(2 * t_gen))) / chi_n < 1.4 + 2 / (dim + 1)
            
            pc = (1 - c_c) * pc + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * v
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            delta = (1 + delta_h) * (1 - c_1)
            
            rank1 = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_norm = y[i] - mean
                rank_mu += weights[i] * np.outer(y_norm, y_norm)
                
            C = delta * C + c_1 * (rank1 - C) + c_mu * rank_mu / (sigma**2)
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * math.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
            
            mean = new_mean

        return best_x, best_f, calls_used
