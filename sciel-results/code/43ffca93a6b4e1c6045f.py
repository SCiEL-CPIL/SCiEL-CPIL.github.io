import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # CMA-ES Parameters
        # Population size
        lam = max(4 + int(3 * np.log(dim)), 10)
        # Initial standard deviation
        sigma = 2.0
        # Initial center
        x_mean = np.zeros(dim)
        
        # CMA-ES specific parameters
        c_sigma = (lam / 2 + 1) / (dim + 4 + lam / 2)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam / 2)
        c_mu = min(c_1, 2 / ((dim + 1.3) ** 2 + 2 * lam / 2))
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1. / (4. * dim) + 1. / (21. * dim ** 2))
        
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Pre-compute Cholesky factor
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        stagnation_counter = 0
        last_best_f = np.inf
        
        while calls < budget - 20 and time.time() < deadline:
            # Generate population
            Y = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.randn(dim)
                Y[i] = x_mean + sigma * (L @ z)
                Y[i] = np.clip(Y[i], -5, 5)
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                if calls >= budget - 20 or time.time() > deadline:
                    break
                F[i] = func(Y[i])
                calls += 1
                update_best(Y[i], F[i])
            
            if calls >= budget - 20 or time.time() > deadline:
                break
            
            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
            
            # Update step-size control (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (L @ (x_new - x_mean) / sigma)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) < (1 + 2 / (dim + 1)) * chi_n
            
            # Update covariance matrix (Rank-Mu update)
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc)
            for i in range(mu):
                C += c_mu * weights[i] * np.outer((Y[i] - x_mean) / sigma, (Y[i] - x_mean) / sigma)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            C += 1e-10 * np.eye(dim)
            
            # Update mean and sigma
            x_mean = x_new
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
                C = np.eye(dim)
            
            # Stagnation check
            if best_f < last_best_f * (1 - 1e-6):
                last_best_f = best_f
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            # If stagnated for too long, restart with a new center?
            # Or just break and polish?
            if stagnation_counter > 10:
                break

        # Polish phase
        if calls < budget and best_x is not None and time.time() < deadline:
            # Use L-BFGS-B or Powell to polish
            # L-BFGS-B requires gradient, which we don't have. So use Powell or Nelder-Mead.
            # Powell is derivative-free and robust.
            try:
                res = minimize(func, best_x, method='Powell', bounds=[(-5, 5)] * dim, 
                               options={'maxiter': 50, 'maxfev': budget - calls, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
