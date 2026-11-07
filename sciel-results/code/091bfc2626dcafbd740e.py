"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with IPOP restarts for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def _evaluate(self, func, x):
        """Evaluate func at x, clip to bounds, track call count and best."""
        x = np.clip(x, -5.0, 5.0)
        fx = float(func(x))
        self.calls += 1
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return x, fx

    def _cma_es_single(self, func, x0, sigma, max_evals, deadline):
        """Run a single CMA-ES instance starting from x0 with initial sigma.
        
        Returns (best_x, best_f) for this run.
        """
        n = self.dim
        # Population size
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        # Weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Covariance matrix
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Step size
        sigma_cur = sigma
        
        # Initial population
        mean = x0.copy()
        
        # Track best within this run
        run_best_x = mean.copy()
        run_best_f = np.inf
        
        # Initial evaluation of mean
        mean_clipped = np.clip(mean, -5.0, 5.0)
        f_mean = float(func(mean_clipped))
        self.calls += 1
        if f_mean < self.best_f:
            self.best_f = f_mean
            self.best_x = mean_clipped.copy()
        if f_mean < run_best_f:
            run_best_f = f_mean
            run_best_x = mean_clipped.copy()
        
        # Precompute cholesky update parameters
        # We'll use a simple approach: recompute Cholesky each step (n=5, so it's fast)
        
        while self.calls < max_evals and time.time() < deadline:
            # Check if we should stop
            if sigma_cur < 1e-15 or sigma_cur > 1e10:
                break
            
            # Generate population
            ys = np.zeros((lam, n))
            xs = np.zeros((lam, n))
            fs = np.zeros(lam)
            
            for i in range(lam):
                y = rng.standard_normal(n)
                ys[i] = y
                x_i = mean + sigma_cur * (C_sqrt @ y)
                x_i = np.clip(x_i, -5.0, 5.0)
                xs[i] = x_i
                f_i = float(func(x_i))
                self.calls += 1
                fs[i] = f_i
                if f_i < self.best_f:
                    self.best_f = f_i
                    self.best_x = x_i.copy()
                if f_i < run_best_f:
                    run_best_f = f_i
                    run_best_x = x_i.copy()
                
                # Check time and budget
                if self.calls >= max_evals or time.time() >= deadline:
                    break
            
            if self.calls >= max_evals or time.time() >= deadline:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = ys[order]
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Update mean
            mean_old = mean.copy()
            mean = np.sum(w[:, None] * xs_sorted[:mu], axis=0)
            
            # Update step size (sigma)
            y_mean = (mean_old - mean) / (sigma_cur)
            norm_y_mean = np.linalg.norm(y_mean)
            
            # Expected norm of standard normal
            chi_n = np.sqrt(n * (1 - 0.75 / (n + 2)))
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            sigma_cur *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance
            y_c = (mean_old - mean) / sigma_cur
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(w[:, None, None] * np.outer(ys_sorted[:mu], ys_sorted[:mu]), axis=0)
            
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * (rank_one - C) + c_mu * (rank_mu - C)
            
            # Recompute Cholesky (with regularization if needed)
            for attempt in range(10):
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt = L
                    C_inv_sqrt = np.linalg.inv(L)
                    break
                except np.linalg.LinAlgError:
                    C += 1e-10 * (attempt + 1) * np.eye(n)
            else:
                # Reset if Cholesky fails repeatedly
                C = np.eye(n)
                C_sqrt = np.eye(n)
                C_inv_sqrt = np.eye(n)
                break
        
        return run_best_x, run_best_f

    def __call__(self, func):
        global rng
        rng = np.random.default_rng(12345)
        
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        # Multi-start CMA-ES with IPOP
        # Start with a few diverse initial points
        n_starts = 5
        x0s = []
        sigmas = []
        
        # First start: center
        x0s.append(np.zeros(self.dim))
        sigmas.append(2.0)
        
        # Additional starts: random
        for i in range(1, n_starts):
            x0 = rng.uniform(-4, 4, self.dim)
            x0s.append(x0)
            sigmas.append(2.0)
        
        # Allocate budget across starts
        # Use IPOP: each restart gets a fraction of the remaining budget
        # and sigma is halved
        
        evals_per_start = max(10, self.budget // (n_starts + 2))
        
        for i in range(n_starts):
            if self.calls >= self.budget or time.time() >= deadline:
                break
            
            remaining = self.budget - self.calls
            if remaining < 10:
                break
            
            # Limit this run's evaluations
            max_evals_this = min(evals_per_start, remaining)
            
            best_x, best_f = self._cma_es_single(
                func, x0s[i], sigmas[i], self.calls + max_evals_this, deadline
            )
            
            # IPOP: halve sigma for next restart
            sigmas[i + 1] = sigmas[i] / 2.0 if i + 1 < n_starts else sigmas[i]
            
            # Optionally, use the best found so far as a new starting point
            if i < n_starts - 1 and self.best_x is not None:
                x0s[i + 1] = self.best_x.copy() + 0.1 * rng.standard_normal(self.dim)
                x0s[i + 1] = np.clip(x0s[i + 1], -4.5, 4.5)
        
        # Final polish: use remaining budget for a local refinement around the best point
        if self.calls < self.budget and time.time() < deadline and self.best_x is not None:
            remaining = self.budget - self.calls
            if remaining >= 5:
                # Local search with small steps
                x = self.best_x.copy()
                fx = self.best_f
                sigma_local = 0.05
                for _ in range(remaining):
                    if time.time() >= deadline:
                        break
                    y = np.clip(x + sigma_local * rng.standard_normal(self.dim), -5.0, 5.0)
                    fy = float(func(y))
                    self.calls += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < self.best_f:
                            self.best_f = fx
                            self.best_x = x.copy()
                        sigma_local *= 1.2
                    else:
                        sigma_local *= 0.8
                    if sigma_local < 1e-6:
                        break
        
        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
