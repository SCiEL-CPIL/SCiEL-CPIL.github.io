"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f16 (Weierstrass) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._eval_count = 0
        self._start_time = None
        self._best_x = None
        self._best_f = np.inf

    def _record(self, x, fx):
        """Record an evaluation and update best."""
        self._eval_count += 1
        if fx < self._best_f:
            self._best_f = fx
            self._best_x = np.array(x, dtype=np.float64)

    def _cmaes_step(self, func, x_mean, sigma, C, pc, ps, step_size, rng):
        """
        Single CMA-ES step. Returns updated parameters and best from this step.
        """
        n = self.dim
        # Compute eigendecomposition of C
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        
        # Sample lambda points
        lam = int(4 + 3 * np.log(n))
        weights = np.log(lam / 2) - np.arange(1, lam + 1)
        weights = np.maximum(weights, 1e-10)
        weights /= weights.sum()
        mu = int(lam / 2)
        nu = 1.0 / (2 * (mu + 0.5))
        
        # Sampling
        z = rng.standard_normal((lam, n))
        # Transform using Cholesky-like: C = V * D * V^T
        # z * V^T * sqrt(D)
        samples = (z @ eigvecs.T * np.sqrt(eigvals)).T  # shape (lam, n)
        candidates = x_mean + sigma * samples
        
        # Evaluate
        fvals = np.array([float(func(candidates[i])) for i in range(lam)])
        for i in range(lam):
            self._record(candidates[i], fvals[i])
        
        # Sort by fitness
        order = np.argsort(fvals)
        selected = candidates[order[:mu]]
        selected_f = fvals[order[:mu]]
        
        # Weighted mean
        w = weights[:mu]
        w = w / w.sum()
        x_new = np.sum(w[:, None] * selected, axis=0)
        
        # Update covariance
        delta = (x_new - x_mean) / sigma
        pc = (1 - step_size) * pc + step_size * delta
        
        # Rank-1 update
        C = (1 + 2 * nu * (np.dot(pc, pc) - 1) - step_size) * C + \
            (1 + 2 * nu - step_size) * np.outer(delta, delta)
        
        # Rank-mu update
        if step_size > 0:
            diff = (selected - x_mean) / sigma
            C = (1 - 2 * nu * (np.dot(pc, pc) - 1) - step_size + step_size * nu) * C
            for i in range(mu):
                C += w[i] * np.outer(diff[i], diff[i])
        
        # Ensure positive semi-definite
        eigvals_new, eigvecs_new = np.linalg.eigh(C)
        eigvals_new = np.maximum(eigvals_new, 1e-20)
        C = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T
        
        # Adaptive sigma
        ps = (1 - step_size) * ps + step_size * (delta / np.sqrt(np.trace(C) / n))
        sigma *= np.exp(0.5 * (np.dot(ps, ps) / n - 1))
        sigma = np.clip(sigma, 1e-10, 10.0)
        
        return x_new, sigma, C, pc, ps, fvals[order[0]], candidates[order[0]]

    def __call__(self, func):
        self._start_time = time.time()
        deadline = 295.0  # safety margin before 330s kill
        
        n = self.dim
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial Sampling
        # Sample 10 points to get a good start
        n_init = 10
        init_points = rng.uniform(-4.5, 4.5, (n_init, n))
        init_fvals = np.array([float(func(p)) for p in init_points])
        for i in range(n_init):
            self._record(init_points[i], init_fvals[i])
        
        best_idx = np.argmin(init_fvals)
        x_mean = init_points[best_idx].copy()
        self._best_x = x_mean.copy()
        self._best_f = init_fvals[best_idx]
        
        if self._eval_count >= self.budget:
            return self._best_x, self._best_f
        
        # Phase 2: CMA-ES with Restarts
        # CMA-ES Parameters
        step_size = 1.0 / (n + 3) ** 1.5
        
        # We will run CMA-ES in chunks. If it stalls, we restart.
        max_evals_per_run = 100
        
        while self._eval_count < self.budget:
            # Check time
            if (time.time() - self._start_time) > deadline:
                break
            
            # Check if we should stop CMA-ES and switch to local polish
            if self.budget - self._eval_count < 50:
                break
                
            # Reset CMA-ES state for this run
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            
            # Start from current best
            x_mean = self._best_x.copy()
            
            # Initial sigma for this run
            # First run: 2.0, subsequent runs (restarts): 0.5
            if self._eval_count == n_init:
                sigma = 2.0
            else:
                sigma = 0.5
                
            # Track stagnation
            best_f_at_start = self._best_f
            evals_at_start = self._eval_count
            
            # Inner loop for this CMA-ES run
            while self._eval_count < self.budget:
                if (time.time() - self._start_time) > deadline:
                    break
                
                # Check budget
                if self._eval_count >= self.budget:
                    break
                    
                # Check stagnation: if no improvement in last 50 evals
                if self._eval_count - evals_at_start > 50:
                    if self._best_f >= best_f_at_start - 1e-12:
                        # Stagnated
                        break
                        
                # Check if sigma is too small
                if sigma < 1e-8:
                    break
                
                try:
                    x_mean, sigma, C, pc, ps, best_f_step, best_x_step = \
                        self._cmaes_step(func, x_mean, sigma, C, pc, ps, step_size, rng)
                except Exception:
                    break
                
                # If we found a better point, update our global best
                if best_f_step < self._best_f:
                    self._best_f = best_f_step
                    self._best_x = best_x_step.copy()
            
            # If we exited the inner loop due to stagnation or sigma small, and we have budget left
            # The outer loop will restart with new parameters
            
            # Safety break to prevent infinite loop if no progress at all
            if self._eval_count == evals_at_start:
                break

        # Phase 3: Local Polish
        remaining = self.budget - self._eval_count
        if remaining > 5 and self._best_x is not None:
            x = self._best_x.copy()
            f = self._best_f
            step_sizes = [0.5, 0.1, 0.01, 0.001, 0.0001]
            
            for ss in step_sizes:
                if self._eval_count >= self.budget or (time.time() - self._start_time) > deadline:
                    break
                improved = True
                while improved and self._eval_count < self.budget:
                    improved = False
                    for i in range(n):
                        if (time.time() - self._start_time) > deadline:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[i] += direction * ss
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = float(func(x_trial))
                            self._record(x_trial, f_trial)
                            if f_trial < f - 1e-12:
                                x = x_trial
                                f = f_trial
                                improved = True
                                break
                        if self._eval_count >= self.budget:
                            break
                    if self._eval_count >= self.budget:
                        break
                
                if f < self._best_f:
                    self._best_f = f
                    self._best_x = x.copy()
        
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
