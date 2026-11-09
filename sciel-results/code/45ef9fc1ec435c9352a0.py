"""
Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Strategy:
1. Sobol-based initialization to quickly locate a promising region.
2. CMA-ES for robust global and local optimization, adapting to the 
   non-separable structure of f14.
3. Strict deadline management to ensure completion within 300s.
"""
import time
import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, calls_counter, deadline, dim):
        """
        Run CMA-ES optimization.
        Returns (best_x, best_f).
        """
        n = dim
        # Population size: standard recommendation is 4 + 3*ln(n)
        lam = 4 + int(3 * np.log(n))
        if lam < 4:
            lam = 4
        mu = lam // 2
        
        # Weights
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / (weights ** 2).sum()
        
        # Strategy parameters
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        c1 = (2 + 2 * mueff / (n + 1) ** 2) / ((n + 2) ** 2 + 2 * mueff / (n + 1) ** 2)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        # State
        mean = x0.copy().astype(float)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        best_x = x0.copy()
        best_f = float(func(x0))
        calls_counter[0] += 1
        
        gen = 0
        while calls_counter[0] < budget and time.time() < deadline:
            # Sample population
            pop = np.zeros((lam, n))
            pop_f = np.zeros(lam)
            
            # Precompute Cholesky for this generation if C hasn't changed significantly?
            # Actually, C changes every gen. We compute it once per gen for sampling.
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset to identity and reduce sigma
                C = np.eye(n)
                sigma *= 0.5
                B = np.linalg.cholesky(C)
            
            for i in range(lam):
                if calls_counter[0] >= budget or time.time() >= deadline:
                    break
                z = np.random.randn(n)
                x_new = mean + sigma * (B @ z)
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = float(func(x_new))
                calls_counter[0] += 1
                pop[i] = x_new
                pop_f[i] = f_new
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            
            if calls_counter[0] >= budget or time.time() >= deadline:
                break
            
            # Sort by fitness
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]
            
            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop[0].copy()
            
            # Weighted recombination
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update evolution paths
            # Inverse Cholesky for path update
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)
            
            pc = (1 - cs) * pc + cs * np.sqrt(mueff) * (C_inv_sqrt @ (mean - mean_old) / sigma)
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * mueff) * (C_inv_sqrt @ (mean - mean_old) / sigma)
            
            # Hessian-free rank-mu and rank-one update
            v = ps / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
            hsig = 1 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - ds) ** (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            # Rank-mu update
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                diff = (pop[i] - mean_old) / sigma
                rank_mu += weights[i] * np.outer(diff, diff)
                
            # Update C
            C = (1 + c1 * hsig * (np.outer(ps, ps) - C) + cmu * rank_mu - (c1 * hsig + cmu) * C)
            
            # Ensure C is symmetric and positive definite
            C = 0.5 * (C + C.T)
            C += 1e-12 * np.eye(n)
            
            # Update sigma (CSA)
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)
            
            gen += 1
            
        return best_x, best_f

    def __call__(self, func):
        start = time.time()
        # 300s limit, keep 10s margin for safety
        deadline = start + 290.0
        dim = self.dim
        budget = self.budget
        calls_counter = [0]
        
        # Phase 1: Sobol Initialization
        # Use a small number of Sobol points to find a good starting point
        # For dim=5, 32 points is a good balance between coverage and cost
        n_init = 32
        if budget < n_init + 10:
            n_init = max(1, budget - 10)
            
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_init)
        # Map [0, 1] to [-5, 5]
        x_init = sobol_points * 10.0 - 5.0
        
        best_x = None
        best_f = np.inf
        
        for i in range(n_init):
            if calls_counter[0] >= budget or time.time() >= deadline:
                break
            x = x_init[i]
            fx = float(func(x))
            calls_counter[0] += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            # Fallback if no points evaluated (should not happen)
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
            calls_counter[0] += 1

        # Phase 2: CMA-ES Refinement
        # Start CMA-ES from the best Sobol point
        # Initial sigma: 1.0 is a reasonable guess for a box of size 10
        # If the function is very flat, it will expand; if steep, it will contract.
        sigma0 = 1.0
        
        # Run CMA-ES with the remaining budget
        best_x, best_f = self._cma_es(
            func, best_x, sigma0, budget, calls_counter, deadline, dim
        )
        
        return best_x, float(best_f)
