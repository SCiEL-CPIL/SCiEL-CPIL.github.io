"""Optimizer using lightweight CMA-ES for BBOB f18 (Schaffers F7 ill-conditioned)."""
import numpy as np
import time

class Optimizer:
    """Lightweight CMA-ES with adaptive restarts for ill-conditioned functions."""
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)
    
    def _cma_es(self, func, x0, sigma, budget, rng, deadline):
        """Run CMA-ES from initial point x0 with step size sigma.
        
        Returns (best_x, best_f, calls_used).
        """
        dim = self.dim
        # Population size: 4 + floor(3 * ln(dim)) for dim=5 -> 4+5=9
        pop_size = 4 + int(3 * np.log(dim)) if dim > 1 else 2
        pop_size = min(pop_size, 12)
        
        mean = x0.copy()
        C = np.eye(dim)
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        
        # CMA-ES parameters
        cc = (4 + dim / 3) / (dim + 4 + 2 / dim)
        cmu = min(1 - cc, 2 / ((dim + 1.3) ** 2 + cc))
        cs = (dim + 1.7) / (dim + 9)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        best_x = mean.copy()
        best_f = func(mean)
        calls = 1
        
        # Weights (fixed, based on pop_size)
        mu = max(1, pop_size // 2)
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / (weights ** 2).sum()
        
        # First generation: generate full population
        while calls < budget:
            if time.time() > deadline - 5.0:
                break
            
            # Generate population
            offspring = []
            for _ in range(pop_size):
                if calls >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                z = rng.standard_normal(dim)
                x_sample = self._clip(mean + sigma * (C_eigvecs @ (C_eigvals ** 0.5 * z)))
                fx = func(x_sample)
                calls += 1
                offspring.append((x_sample, fx))
                if fx < best_f:
                    best_f = fx
                    best_x = x_sample.copy()
            
            if len(offspring) < 2:
                break
            
            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            lambda_ = len(offspring)
            mu_actual = min(mu, lambda_)
            weights_actual = weights[:mu_actual]
            mueff_actual = 1 / (weights_actual ** 2).sum()
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights_actual[:, None] * np.array([o[0] for o in offspring[:mu_actual]]), axis=0)
            
            # Update evolution paths
            y = (mean - old_mean) / sigma if sigma > 1e-15 else np.zeros(dim)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff_actual) * y
            if np.linalg.norm(ps) / chi_n < (1.4 + cc) * np.sqrt(1 + (1 - cs) / cs):
                pc = (1 - cc) * pc + cc * y
            
            # Update covariance matrix (correct formula)
            C = (1 - ds - cmu * mueff_actual) * C + \
                (ds - 2 + cmu * mueff_actual) * np.outer(pc, pc) + \
                cmu * sum(w_i * np.outer((offspring[i][0] - old_mean) / sigma, (offspring[i][0] - old_mean) / sigma) for i, w_i in enumerate(weights_actual))
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Eigendecomposition
            try:
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                C_eigvals = np.clip(C_eigvals, 1e-20, None)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_eigvals = np.ones(dim)
                C_eigvecs = np.eye(dim)
            
            # Sigma adaptation (CSA)
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 20.0)
            
            # Check for convergence
            if np.linalg.norm(mean - old_mean) < 1e-8 * sigma:
                break
        
        return best_x, best_f, calls
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Use most of the budget for a single CMA-ES run from origin
        # with a reasonable initial sigma for [-5,5]^5
        sigma_init = 2.0
        x0 = np.zeros(dim)
        
        # Reserve a small amount for potential second run
        first_budget = budget - 50 if budget > 100 else budget
        
        x_cma, f_cma, calls_cma = self._cma_es(func, x0, sigma_init, first_budget, rng, deadline)
        total_calls = calls_cma
        best_f = f_cma
        best_x = x_cma.copy()
        
        # If we have remaining budget and time, run a second CMA-ES from a different start
        if total_calls < budget - 10 and time.time() < deadline - 10.0:
            # Use the best found so far as starting point for refinement
            remaining = budget - total_calls
            sigma_second = max(0.5, sigma_init * 0.3)
            x_cma2, f_cma2, calls_cma2 = self._cma_es(func, best_x, sigma_second, remaining, rng, deadline)
            total_calls += calls_cma2
            if f_cma2 < best_f:
                best_f = f_cma2
                best_x = x_cma2.copy()
        
        return best_x, best_f
