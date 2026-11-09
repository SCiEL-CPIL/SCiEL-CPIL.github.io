"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-stage optimizer: LHS spread + CMA-ES + L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _lhs_sample(self, n, dim, rng):
        """Latin Hypercube Sampling."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return samples * 10.0 - 5.0  # Map to [-5, 5]

    def _cmaes(self, func, x0, sigma, budget, deadline, dim):
        """Simple CMA-ES implementation."""
        rng = np.random.default_rng(42)
        x = x0.copy()
        fx = func(x)
        C = np.eye(dim)
        step_size = sigma
        
        # CMA-ES parameters
        mu = max(1, dim // 2)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / (dim + 2)) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        calls_made = 0
        best_x = x.copy()
        best_f = fx
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            population = []
            for i in range(mu):
                z = rng.standard_normal(dim)
                y = x + step_size * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                population.append((y, fy))
            
            # Sort by fitness
            population.sort(key=lambda p: p[1])
            
            # Update best
            for y, fy in population:
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            # Compute weighted mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * population[i][0]
            
            # Update covariance matrix
            diff = x_new - x
            x = x_new
            
            # Update step size
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-10:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * diff / step_size
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls_made + 1))) / chi_n < 1.4 + 2 / (dim + 1)
                if h_sigma:
                    p_c = (1 - c_c) * p_c + c_c * diff / step_size
                else:
                    p_c = (1 - c_c) * p_c
                C = (1 + c_1 * (h_sigma - 1) + c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(population[i][0] - x_new, population[i][0] - x_new) for i in range(mu))
                step_size *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Evaluate new center if budget allows
            if calls_made < budget:
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        return best_x, best_f, calls_made

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Stage 1: LHS spread for early progress
        n_lhs = 31
        lhs_samples = self._lhs_sample(n_lhs, dim, rng)
        # Include origin
        lhs_samples = np.vstack([lhs_samples, np.zeros(dim)])
        
        best_x = lhs_samples[0].copy()
        best_f = func(lhs_samples[0])
        calls_made = 1
        
        for i in range(1, len(lhs_samples)):
            if time.time() > deadline:
                break
            fi = func(lhs_samples[i])
            calls_made += 1
            if fi < best_f:
                best_f = fi
                best_x = lhs_samples[i].copy()
        
        # Stage 2: CMA-ES from best LHS point
        remaining_budget = budget - calls_made
        cma_budget = int(remaining_budget * 0.7)
        
        if cma_budget > 10:
            x_cma, f_cma, cma_calls = self._cmaes(func, best_x, 1.0, cma_budget, deadline, dim)
            calls_made += cma_calls
            if f_cma < best_f:
                best_f = f_cma
                best_x = x_cma.copy()
        
        # Stage 3: L-BFGS-B polish if time and budget remain
        remaining_budget = budget - calls_made
        if remaining_budget > 5 and time.time() < deadline - 30:
            try:
                from scipy.optimize import minimize
                result = minimize(func, best_x, method='L-BFGS-B', bounds=[(-5, 5)] * dim,
                                  options={'maxiter': 100, 'maxfun': remaining_budget})
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
