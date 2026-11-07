"""Population-based CMA-ES for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for multimodal Weierstrass function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, dim, budget_remaining, start_time, deadline, rng, initial_mean=None):
        """Run CMA-ES with given budget. Returns (best_x, best_f, calls_made)."""
        if initial_mean is None:
            mean = rng.uniform(-5, 5, dim)
        else:
            mean = initial_mean.copy()
        
        # Initial step size
        sigma = 2.0
        
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # ~13 for dim=5
        mu = lambda_ // 2  # number of recombination weights
        
        # Recombination weights
        weights = np.log(np.arange(lambda_ / 2 + 1.0)) - np.log(np.arange(lambda_ / 2 + 1.0, 0, -1.0))
        weights = weights[:mu]
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))
        
        # State variables
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))
        
        best_x = mean.copy()
        best_f = np.inf
        calls_made = 0
        
        while calls_made < budget_remaining:
            if time.time() > deadline:
                break
            
            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Cholesky decomposition for sampling
                C_cholesky = np.linalg.cholesky(C)
                pop[i] = mean + sigma * (C_cholesky @ z)
            
            # Evaluate population
            fvals = np.array([func(x) for x in pop])
            calls_made += lambda_
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = pop[min_idx].copy()
            
            # Sort by fitness
            indices = np.argsort(fvals)
            y_sorted = pop[indices]
            fvals_sorted = fvals[indices]
            
            # Weighted recombination
            y_mean = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ y_mean)
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lambda_ + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            mean += sigma * y_mean
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            delta_heav = (1 - c_c) ** (2 * (calls_made // lambda_ + 1)) - 1
            rank_mu = np.outer(y_mean, y_mean)
            C = (1 + c_1 * (np.dot(pc, pc) - 1 + delta_heav) + c_mu * (rank_mu - np.outer(pc, pc)) - c_1) * C
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2
            
            # Clip mean to bounds
            mean = np.clip(mean, -5, 5)
            
            # Check for stagnation
            if calls_made > 50 and np.all(np.abs(np.diff(fvals_sorted[:5])) < 1e-10):
                break
        
        return best_x, best_f, calls_made

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Restart strategy: multiple short CMA-ES runs
        restart_budgets = [150, 100, 80, 60, 40, 30, 20, 10, 5, 5]
        
        for i, rb in enumerate(restart_budgets):
            if total_calls >= budget or time.time() > deadline:
                break
            
            remaining = min(rb, budget - total_calls)
            if remaining < 10:
                break
            
            # Initial mean: first run random, subsequent runs near best or random
            if i == 0:
                init_mean = None
            else:
                # Mix of best-so-far and random restart
                if rng.random() < 0.5:
                    init_mean = best_x + rng.standard_normal(dim) * 0.5
                    init_mean = np.clip(init_mean, -5, 5)
                else:
                    init_mean = rng.uniform(-5, 5, dim)
            
            bx, bf, calls = self._cma_es(func, dim, remaining, start_time, deadline, rng, init_mean)
            total_calls += calls
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
        
        # If we have budget left, do a final local refinement
        if total_calls < budget and best_x is not None and time.time() < deadline:
            # Simple local search around best
            step = 0.1
            for _ in range(20):
                if total_calls >= budget or time.time() > deadline:
                    break
                x_new = best_x + rng.standard_normal(dim) * step
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                total_calls += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    step *= 0.8
                else:
                    step *= 0.9
        
        return best_x, best_f
# EVOLVE-BLOCK-END
