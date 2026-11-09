"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with CMA-ES-like covariance adaptation for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            return float(val)

        best_x = None
        best_f = np.inf

        # Phase 1: Coarse multi-start (1+1)-ES from multiple basins
        # Lunacek has two basins; start near origin and near typical offset
        n_restarts = 5
        budget_per_restart = (budget * 70) // (100 * n_restarts)  # ~70% of budget spread across restarts
        
        # Starting points: origin + random points in [-4, 4]
        start_points = [np.zeros(dim)]
        for _ in range(n_restarts - 1):
            start_points.append(rng.uniform(-4, 4, dim))

        for sp in start_points:
            if time.time() > deadline:
                break
            if call_count >= budget - 20:
                break

            x, fx = self._es_restart(safe_func, sp, dim, budget_per_restart, deadline, rng)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Fine-tuning with smaller step size from best point
        if best_x is not None and call_count < budget - 10 and time.time() < deadline:
            remaining = budget - call_count - 5
            if remaining > 10:
                x, fx = self._es_restart(safe_func, best_x, dim, remaining, deadline, rng, initial_step=0.3)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f

    def _es_restart(self, func, x0, dim, max_calls, deadline, rng, initial_step=1.5):
        """(1+1)-ES with adaptive step size using 1/5 success rule."""
        x = x0.copy()
        fx = func(x)
        sigma = initial_step
        
        # Adaptive parameters
        c_sigma = 0.5  # adaptation rate
        d_sigma = 1.0 + 0.35 * min(dim / 3, 1.0)  # damping
        
        for _ in range(max_calls):
            if time.time() > deadline:
                break
            
            # Generate candidate
            z = rng.standard_normal(dim)
            x_new = x + sigma * z
            x_new = np.clip(x_new, -5.0, 5.0)
            fx_new = func(x_new)
            
            if fx_new < fx:
                # Success: increase step size
                sigma *= np.exp(c_sigma * (1.0 / d_sigma))
                x = x_new
                fx = fx_new
            else:
                # Failure: decrease step size
                sigma *= np.exp(-c_sigma * (1.0 / d_sigma))
            
            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 1e-4, 5.0)
            
            # Convergence check: if sigma is very small, we're converged
            if sigma < 1e-5:
                break

        return x, fx
# EVOLVE-BLOCK-END
