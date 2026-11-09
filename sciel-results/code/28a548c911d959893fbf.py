import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf
        self._start_time = None
        self._deadline = None

    def __call__(self, func):
        self._start_time = time.time()
        # Leave 5s margin for safety
        self._deadline = self._start_time + 325
        self._call_count = 0
        self._best_x = np.zeros(self.dim)
        self._best_f = np.inf
        self._rng = np.random.default_rng(42)

        def safe_call(x):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Phase 1: Global Exploration using Sobol Sequence
        # Sobol provides low-discrepancy points, ensuring good coverage
        n_global = 100
        try:
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            sobol_points = sampler.random(n_global)
            # Map [0,1]^dim to [-5,5]^dim
            sobol_points = sobol_points * 10.0 - 5.0
        except Exception:
            # Fallback to random if Sobol fails
            sobol_points = self._rng.uniform(-5, 5, (n_global, self.dim))

        # Evaluate global points
        f_values = np.zeros(n_global)
        for i in range(n_global):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            f_values[i] = safe_call(sobol_points[i])
        
        # If we didn't evaluate all, pad with inf
        if self._call_count < n_global:
             f_values[self._call_count:] = 1e10

        # Find best point from global phase
        best_idx = np.argmin(f_values)
        x_start = sobol_points[best_idx].copy()
        f_start = f_values[best_idx]
        
        if f_start >= 1e9:
             # Fallback if all failed
             x_start = np.zeros(self.dim)
             f_start = safe_call(x_start)

        # Phase 2: Local Optimization using CMA-ES
        # CMA-ES is robust for this dimensionality and budget
        if self._call_count < self.budget and time.time() < self._deadline:
            self._run_cmaes(func, safe_call, x_start)

        return self._best_x, self._best_f

    def _run_cmaes(self, func, safe_call, x0):
        dim = self.dim
        # Population size
        lambda_ = max(10, int(4 + 3 * np.log(dim)))
        
        # CMA-ES parameters
        sigma = 2.0  # Initial step size
        mean = x0.copy()
        
        # Covariance matrix C (initially identity)
        C = np.eye(dim)
        
        # Eigenvalue decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Weights for mean update
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        # Step size control
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        ps = np.zeros(dim)
        
        # Damping
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Generation loop
        while self._call_count < self.budget and time.time() < self._deadline:
            # Generate population
            # We need to sample from N(0, C)
            # x = mean + sigma * eigvecs @ (sqrt(eigvals) * z)
            
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                if self._call_count >= self.budget or time.time() >= self._deadline:
                    break
                z = self._rng.standard_normal(dim)
                # Transform z to have covariance C
                # C = eigvecs @ diag(eigvals) @ eigvecs.T
                # sqrt(C) z = eigvecs @ (sqrt(eigvals) * z)
                sample = eigvecs @ (np.sqrt(eigvals) * z)
                pop[i] = mean + sigma * sample
                pop[i] = np.clip(pop[i], -5.0, 5.0)
            
            # If we broke out early, we can't do a full generation
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
                
            # Evaluate population
            f_pop = np.zeros(lambda_)
            for i in range(lambda_):
                f_pop[i] = safe_call(pop[i])
            
            # Sort population
            idx = np.argsort(f_pop)
            pop = pop[idx]
            f_pop = f_pop[idx]
            
            # Update mean
            new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Update covariance matrix
            # Calculate step for covariance update
            y = (new_mean - mean) / sigma
            
            # Update ps (step size control)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (eigvecs.T @ y) / np.sqrt(np.diag(C))
            
            # Update C
            delta_hsig = (1.04 + 2 / (dim + 1)) * (np.linalg.norm(ps) / chi_n - 1)
            
            C = (1 - c_c - c_1 - c_mu + delta_hsig * c_sigma * c_sigma) * C \
                + c_1 * np.outer(y, y) \
                + c_mu / sigma**2 * np.sum(weights[:, None, None] * np.outer(pop[:mu] - mean, pop[:mu] - mean), axis=0)
            
            # Update sigma
            sigma *= np.exp(c_sigma / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            mean = new_mean
            
            # Re-decompose C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure eigvals are positive (numerical stability)
            eigvals = np.clip(eigvals, 1e-12, None)
            
            # Check for convergence/stagnation
            if sigma < 1e-6:
                break

        # Optional: Final polish with (1+1)-ES if budget remains
        if self._call_count < self.budget and time.time() < self._deadline:
            sigma = max(0.1, 0.5 * (self.budget - self._call_count) / self.budget)
            for i in range(50):
                if self._call_count >= self.budget or time.time() >= self._deadline:
                    break
                noise = self._rng.standard_normal(dim)
                candidate_x = self._best_x + sigma * noise
                candidate_x = np.clip(candidate_x, -5.0, 5.0)
                f_cand = safe_call(candidate_x)
                if f_cand < self._best_f:
                    self._best_f = f_cand
                    self._best_x = candidate_x
                    sigma *= 0.95
                else:
                    sigma *= 0.9
