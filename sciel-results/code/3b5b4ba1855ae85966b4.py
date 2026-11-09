"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES + (1+1)-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, fx0, budget_cma, rng):
        """Run CMA-ES for a given number of function evaluations."""
        n = self.dim
        lam = max(4, int(np.ceil(4 + 3 * np.log(n))))
        
        # CMA-ES parameters
        c_sigma = (n + 2) / (n + 4 + 2 * n / n)
        c_1 = (2 + 2.0 / (n + 1.3)) / ((n + 2)**2 + 2)
        c_mu = min(1 - c_1, (2 * (mu_eff - 2) / ((n + 2)**2 + 2)))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Recalculate c_mu with correct mu_eff
        c_mu = min(1 - c_1, (2 * (mu_eff - 2) / ((n + 2)**2 + 2)))
        
        sigma = 2.0
        mean = x0.copy()
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Precompute eigendecomposition update
        def update_covariance(C, C_inv_sqrt, C_inv):
            # Use SVD for numerical stability
            try:
                d, V = np.linalg.eigh(C)
                d = np.clip(d, 1e-12, None)
                C_inv_sqrt = V @ np.diag(1.0 / np.sqrt(d)) @ V.T
                C_inv = V @ np.diag(1.0 / d) @ V.T
            except:
                C_inv_sqrt = np.linalg.inv(np.sqrtm(C + 1e-10 * np.eye(n)))
                C_inv = np.linalg.inv(C + 1e-10 * np.eye(n))
            return C_inv_sqrt, C_inv
        
        C_inv_sqrt, C_inv = update_covariance(C, C_inv_sqrt, None)
        
        calls_used = 0
        best_x, best_fx = mean.copy(), fx0
        
        while calls_used < budget_cma:
            # Generate lambda offspring
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                z = rng.standard_normal(n)
                y = mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append(y)
                offspring_f.append(fy)
                
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            
            # Sort offspring
            idx = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in idx]
            sorted_f = [offspring_f[i] for i in idx]
            
            # Update mean
            new_mean = np.sum(weights[:, None] * np.array(sorted_offspring[:mu]), axis=0)
            
            # Update step size control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (new_mean - mean) / sigma)
            hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(calls_used + 1)) / chi_n) < 1.4 + 2 / (n + 1)
            
            if hs:
                pc = (1 - c_1) * pc + np.sqrt(c_1 * (2 - c_1) * mu_eff) * (new_mean - mean) / sigma
            else:
                pc = (1 - c_1) * pc
            
            # Update covariance
            diff = np.array(sorted_offspring[:mu]) - mean
            delta_hs = (1.0 if hs else 0.0) - (1 - c_1)
            C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + \
                np.sum(weights[:, None, None] * np.array([np.outer(d, d) for d in diff]), axis=0) * c_mu / mu_eff
            
            # Symmetrize
            C = 0.5 * (C + C.T)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            mean = new_mean
            C_inv_sqrt, C_inv = update_covariance(C, C_inv_sqrt, C_inv)
            
            # Safety: if sigma becomes too small or too large, break
            if sigma < 1e-10 or sigma > 100:
                break
        
        return best_x, best_fx, calls_used

    def _es_11(self, func, x0, fx0, budget_es, rng):
        """Run (1+1)-ES for fine-tuning."""
        x = x0.copy()
        fx = fx0
        sigma = 0.1
        calls_used = 0
        
        for _ in range(budget_es):
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.9
            if sigma < 1e-8:
                break
        
        return x, fx, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        
        # Phase 1: CMA-ES for 60% of budget
        budget_cma = int(self.budget * 0.6)
        budget_es = self.budget - budget_cma
        
        best_x, best_fx, cma_calls = self._cma_es(func, best_x, best_fx, budget_cma, rng)
        
        # Phase 2: (1+1)-ES for remaining budget
        if time.time() < deadline - 5:
            best_x, best_fx, es_calls = self._es_11(func, best_x, best_fx, budget_es, rng)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
