"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style population-based optimizer with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        # Total budget is 500 function calls
        total_budget = self.budget
        dim = self.dim

        # We'll do a few restarts of CMA-ES
        # Each restart gets a portion of the budget
        # Let's do 3 restarts with ~166 calls each, or adapt based on convergence

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            if calls_used >= total_budget:
                return np.inf
            x = np.clip(x, -5, 5)
            calls_used += 1
            try:
                f = float(func(x))
                if np.isfinite(f):
                    return f
                return np.inf
            except Exception:
                return np.inf

        # CMA-ES implementation
        def cma_es(x0, sigma0, budget_for_this_run, rng):
            n = dim
            # Larger population for better exploration of multimodal landscape
            lam = max(8, int(4 + 3 * np.log(n)) + 4)
            
            # CMA-ES parameters
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            
            # Learning rates
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            # Initial state
            mean = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            # Precompute
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            
            # Track best
            best_x_local = mean.copy()
            best_f_local = np.inf
            
            # Convergence tracking
            no_improve = 0
            prev_best = np.inf
            
            for gen in range(budget_for_this_run // lam + 2):
                if time.time() > deadline:
                    break
                if calls_used >= total_budget:
                    break
                
                # Generate population
                # Use Cholesky of C for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                
                # Generate lam samples
                ys = rng.standard_normal((lam, n))
                xs = np.array([mean + sigma * (ys[i] @ L) for i in range(lam)])
                
                # Evaluate
                fs = np.array([safe_func(xs[i]) for i in range(lam)])
                
                # Track best
                idx_min = np.argmin(fs)
                if fs[idx_min] < best_f_local:
                    best_f_local = fs[idx_min]
                    best_x_local = xs[idx_min].copy()
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Early stop if converged
                if no_improve > 15:
                    break
                
                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = ys[order]
                
                # Update mean with weighted recombination
                mean_old = mean.copy()
                mean = sum(weights[i] * xs[order[i]] for i in range(mu))
                
                # Update step-size control (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (L.T @ (mean - mean_old))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 100.0)
                
                # Update covariance matrix
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) / sigma * (mean - mean_old)
                
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                    + c1 * (np.outer(pc, pc) - C) \
                    + cmu * sum(weights[i] * np.outer(ys_sorted[i], ys_sorted[i]) for i in range(mu))
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Add small regularization if needed
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-8 * np.eye(n)
            
            return best_x_local, best_f_local

        # Strategy: Adaptive restarts with convergence-based stopping
        # More restarts with smaller budgets to explore different basins

        rng = np.random.default_rng(42)
        
        # Initial diverse sampling to find good starting points
        # Use 10 calls for initial exploration
        init_calls = min(10, total_budget)
        init_points = []
        init_fvals = []
        for i in range(init_calls):
            x = rng.uniform(-4.5, 4.5, dim)
            f = safe_func(x)
            init_points.append(x)
            init_fvals.append(f)
        
        # Find the best initial point
        if init_fvals:
            best_init_idx = np.argmin(init_fvals)
            best_init_x = init_points[best_init_idx]
            best_init_f = init_fvals[best_init_idx]
        else:
            best_init_x = np.zeros(dim)
            best_init_f = np.inf
        
        if best_init_f < best_f:
            best_f = best_init_f
            best_x = best_init_x
        
        # Now do CMA-ES restarts from best initial points
        # Sort initial points by fitness
        sorted_indices = np.argsort(init_fvals)
        
        # Do up to 4 restarts, each with a portion of remaining budget
        remaining_budget = total_budget - calls_used
        num_restarts = min(4, max(1, remaining_budget // 50))
        budget_per_restart = remaining_budget // num_restarts if num_restarts > 0 else 0
        
        for restart_idx in range(num_restarts):
            if calls_used >= total_budget or time.time() > deadline:
                break
            
            # Choose starting point: best from initial, or random for later restarts
            if restart_idx < len(sorted_indices) and init_fvals[sorted_indices[restart_idx]] < np.inf:
                x0 = init_points[sorted_indices[restart_idx]].copy()
                sigma0 = 1.5
            else:
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 2.5 + restart_idx * 0.5
            
            budget_for_this = min(budget_per_restart, total_budget - calls_used)
            if budget_for_this < 20:
                break
            
            x_r, f_r = cma_es(x0, sigma0, budget_for_this, rng)
            if f_r < best_f:
                best_f = f_r
                best_x = x_r
        
        if best_x is None:
            best_x = np.zeros(dim)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
