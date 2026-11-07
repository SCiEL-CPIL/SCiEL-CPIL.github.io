import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng):
        """Run CMA-ES from x0 with initial sigma, using at most budget function calls."""
        dim = self.dim
        n = dim
        # CMA-ES parameters
        mu = max(1, n // 2)  # number of parents
        lambda_ = 2 * mu  # population size
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff / (n + 2)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        
        # Initial values
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Initial population
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        
        while calls < budget:
            if time.time() - self._start_time > self._time_limit:
                break
            
            # Generate population
            population = np.zeros((lambda_, n))
            for i in range(lambda_):
                z = rng.standard_normal(n)
                # Cholesky decomposition of C
                L = np.linalg.cholesky(C)
                population[i] = x + sigma * (L @ z)
            
            # Evaluate
            fitness = np.zeros(lambda_)
            for i in range(lambda_):
                if calls >= budget:
                    break
                if time.time() - self._start_time > self._time_limit:
                    break
                fitness[i] = func(population[i])
                calls += 1
            
            # Sort by fitness
            idx = np.argsort(fitness)
            population = population[idx]
            fitness = fitness[idx]
            
            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = population[0].copy()
            
            # Update mean
            x_new = np.sum(w[:, None] * population[:mu], axis=0)
            
            # Update covariance matrix
            diff = (x_new - x) / sigma
            pc = (1 - c_c) * pc + c_c * diff
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            
            # Rank-one and rank-mu updates
            delta_sig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) - 1) * c_sigma
            C = (1 - c_c * (1 - c_c) - c_sigma * (2 - c_sigma) * mu_eff) * C \
                + c_c * np.outer(pc, pc) \
                + c_sigma * mu_eff / sigma**2 * np.sum(
                    w[:, None, None] * np.outer(population[:mu] - x, population[:mu] - x), axis=0
                )
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Eigenvalue correction for numerical stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            x = x_new
            sigma *= np.exp(delta_sig)
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 50)
        
        return best_x, best_f, calls

    def __call__(self, func):
        self._start_time = time.time()
        self._time_limit = 285  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                raise Exception("Budget exceeded")
            if time.time() - self._start_time > self._time_limit:
                raise TimeoutError("Time limit exceeded")
            
            calls += 1
            
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
                
            return val

        # Multi-restart CMA-ES strategy
        # First restart: broad search
        rng = np.random.default_rng(42)
        
        # Initial point: center of the box
        x0 = np.zeros(self.dim)
        sigma0 = 2.0  # Large initial step for global search
        
        # Use ~70% of budget for first CMA-ES run
        first_budget = int(0.7 * self.budget)
        
        try:
            bx, bf, used = self._cma_es(wrapped_func, x0, sigma0, first_budget, rng)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        except (TimeoutError, Exception):
            pass

        # Second restart: refine around best found
        remaining = self.budget - calls
        if remaining > 10 and best_x is not None:
            # Start near the best point with smaller sigma
            x0_2 = best_x.copy()
            sigma0_2 = 0.3
            
            # Use remaining budget for refinement
            try:
                bx2, bf2, used2 = self._cma_es(wrapped_func, x0_2, sigma0_2, remaining, rng)
                if bf2 < best_f:
                    best_f = bf2
                    best_x = bx2.copy()
            except (TimeoutError, Exception):
                pass
        
        # Third restart: another refinement with different perturbation
        remaining = self.budget - calls
        if remaining > 10 and best_x is not None:
            x0_3 = np.clip(best_x + 0.2 * rng.standard_normal(self.dim), -5, 5)
            sigma0_3 = 0.1
            
            try:
                bx3, bf3, used3 = self._cma_es(wrapped_func, x0_3, sigma0_3, remaining, rng)
                if bf3 < best_f:
                    best_f = bf3
                    best_x = bx3.copy()
            except (TimeoutError, Exception):
                pass

        return best_x, best_f
