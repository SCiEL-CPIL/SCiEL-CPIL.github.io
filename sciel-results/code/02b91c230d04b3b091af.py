"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES for BBOB f4 (Büche-Rastrigin) on [-5,5]^5.
    
    Uses a portfolio of CMA-ES runs with staggered starts to balance
    early exploration (for AOCC) with late exploitation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes_step(self, x, sigma, C, pc, ps, func, rng, budget_used, max_evals, 
                    min_sigma=1e-8, max_sigma=100.0, cc=None, cs=None, cmu=None, 
                    de=5, time_limit=None, start_time=None):
        """Perform CMA-ES iterations until budget or time runs out.
        
        Returns (x_best, f_best, calls_made, x, sigma, C, pc, ps)
        """
        n = self.dim
        if cc is None:
            cc = (4 + n / 3) / (n + 4 + 2 * n / 3)
        if cs is None:
            cs = (1 + 2 * (n - 2) / (n + 1)) / (n + 4 + 2 * (n - 2) / (n + 1))
        if cmu is None:
            cmu = 2 / ((n + 4.1)**2 + n)
        
        lambda_ = 4 + int(3 * np.log(n))  # population size
        mu = lambda_ // 2  # number of parents
        
        x_best = x.copy()
        f_best = func(x)
        calls = 1
        
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                return x_best, f_best, calls, x, sigma, C, pc, ps
        
        while calls < max_evals:
            if time_limit is not None and start_time is not None:
                if time.time() - start_time > time_limit:
                    break
            
            # Generate population
            y = np.zeros((lambda_, n))
            fy = np.zeros(lambda_)
            for i in range(lambda_):
                y[i] = x + sigma * (C @ rng.standard_normal(n))
                y[i] = np.clip(y[i], -5, 5)
                fy[i] = func(y[i])
                calls += 1
                if calls >= max_evals:
                    break
            
            # Sort by fitness
            idx = np.argsort(fy)
            y = y[idx]
            fy = fy[idx]
            
            # Update best
            if fy[0] < f_best:
                f_best = fy[0]
                x_best = y[0].copy()
            
            # Weights
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(mu + 0.5) - np.log(i + 1)
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w**2)
            
            # Update mean
            x_old = x.copy()
            x = x + sum(w[i] * y[i] for i in range(mu))
            
            # Update path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / np.linalg.sqrt(np.sum(np.diag(C))) * (C @ (w @ y))
            hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / np.sqrt(1 - (1 - cs)**(2 * calls)) < (1.4 + 2 / (n + 1)) * np.trace(C) / n else 0.0
            pc = (1 - cc) * pc + hsig * cc * (x - x_old) / sigma
            
            # Update covariance
            C = (1 + cmu * (hsig * (1 + cc - cc) - cc) * np.trace(C) - cmu) * C
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = C + cmu * (np.outer(pc, pc) + delta_hsig * C)
            for i in range(mu):
                z = C @ (y[i] - x_old) / sigma
                C += cmu * w[i] * (np.outer(z, z) - C)
            
            # Update sigma
            sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ps) / np.sqrt(2 * n * (1 - np.exp(-2 * (calls + 1) / (n + 1)))) - 1))
            sigma = max(min(sigma, max_sigma), min_sigma)
            
            # Ensure C is symmetric and positive definite
            C = 0.5 * (C + C.T)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 1e-10):
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
            except:
                C = np.eye(n)
        
        return x_best, f_best, calls, x, sigma, C, pc, ps

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        
        # Phase 1: Quick random sampling for early progress (first ~10% of budget)
        n_early = min(50, self.budget // 10)
        x_best = np.zeros(self.dim)
        f_best = np.inf
        calls = 0
        
        for i in range(n_early):
            if time.time() - start_time > time_limit:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
        
        # Phase 2: Multi-start CMA-ES
        # Run several CMA-ES instances from different starting points
        # Allocate remaining budget among them
        
        remaining_budget = self.budget - calls
        if remaining_budget <= 0:
            return x_best, f_best
        
        # Number of CMA-ES runs based on budget
        n_runs = min(3, max(1, remaining_budget // 100))
        budget_per_run = remaining_budget // n_runs
        
        # Starting points: mix of best found and random
        starts = [x_best.copy()]
        for i in range(n_runs - 1):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        for run_idx in range(n_runs):
            if time.time() - start_time > time_limit:
                break
            if calls >= self.budget:
                break
            
            x_start = starts[run_idx]
            sigma_start = 2.0
            C = np.eye(self.dim)
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            
            max_evals_for_run = min(budget_per_run, self.budget - calls)
            
            x_b, f_b, c, x, sigma, C, pc, ps = self._cmaes_step(
                x_start, sigma_start, C, pc, ps, func, rng, 
                calls, max_evals_for_run, time_limit=time_limit, start_time=start_time
            )
            calls += c
            
            if f_b < f_best:
                f_best = f_b
                x_best = x_b.copy()
        
        # Phase 3: Final local refinement around best point
        remaining = self.budget - calls
        if remaining > 5 and time.time() - start_time < time_limit:
            # Simple local search with decreasing step
            x_cur = x_best.copy()
            f_cur = f_best
            step = 0.5
            for i in range(min(remaining, 30)):
                if time.time() - start_time > time_limit:
                    break
                if calls >= self.budget:
                    break
                y = x_cur + step * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                if fy < f_cur:
                    x_cur, f_cur = y, fy
                    step *= 1.3
                    if f_cur < f_best:
                        f_best = f_cur
                        x_best = x_cur.copy()
                else:
                    step *= 0.7
        
        return x_best, f_best
# EVOLVE-BLOCK-END
