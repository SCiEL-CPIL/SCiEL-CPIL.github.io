"""Rewritten optimizer: multi-start CMA-ES with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        def safe_call(x):
            nonlocal total_calls
            if total_calls >= budget:
                return None
            total_calls += 1
            try:
                val = func(x)
                if np.isfinite(val):
                    return val
            except Exception:
                pass
            return None

        # Multi-start CMA-ES
        # We'll use a simplified CMA-ES implementation
        # Key idea: population-based evolution with covariance matrix adaptation
        
        n_starts = 5  # number of restarts
        calls_per_start = budget // n_starts
        
        for start_idx in range(n_starts):
            if total_calls >= budget or not time_ok():
                break
            
            # Initialize around origin with some spread
            # Rastrigin has local minima at integers, so start near 0
            if start_idx == 0:
                x0 = np.zeros(dim)
            else:
                # Random restart with moderate spread
                x0 = rng.uniform(-3, 3, dim)
            
            fx0 = safe_call(x0)
            if fx0 is None:
                continue
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
            
            if total_calls >= budget or not time_ok():
                break
            
            # CMA-ES parameters
            pop_size = max(4 + int(3 * np.log(dim)), 8)
            # Adjust population size based on remaining budget
            remaining = budget - total_calls
            if remaining < pop_size:
                break
            
            # Limit population size to not exceed remaining budget
            pop_size = min(pop_size, max(4, remaining // 3))
            
            # Initial step size
            sigma = 1.0
            # CMA-ES parameters
            cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + 4 + dim / (dim + 4))
            cs = (dim + 1.7) / (dim + 3.2 + cs_ if False else 1.0)  # will fix below
            cs = (dim + 1.7) / (dim + 3.2 + 1.0)
            cmu = min(1 - cc, 2 / ((dim + 1.7)**2 + cs))
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            # CMA-ES state
            mean = x0.copy()
            C = np.eye(dim)
            Cinv = np.eye(dim)
            Csqrt = np.eye(dim)
            Cinv_sqrt = np.eye(dim)
            step_size = sigma
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Precompute some values
            def update_chol(C):
                try:
                    L = np.linalg.cholesky(C)
                    return L
                except:
                    # Regularize
                    C_reg = C + 1e-10 * np.eye(dim)
                    L = np.linalg.cholesky(C_reg)
                    return L
            
            Csqrt = update_chol(C)
            
            max_iter = min(remaining, 50)  # limit iterations per start
            for it in range(max_iter):
                if total_calls + pop_size > budget or not time_ok():
                    break
                
                # Generate population
                Y = np.zeros((pop_size, dim))
                for i in range(pop_size):
                    if not time_ok():
                        break
                    z = rng.standard_normal(dim)
                    y = step_size * Csqrt @ z
                    x = np.clip(mean + y, -5, 5)
                    Y[i] = x
                
                # Evaluate population
                F = np.full(pop_size, np.inf)
                for i in range(pop_size):
                    if not time_ok():
                        break
                    fi = safe_call(Y[i])
                    if fi is not None:
                        F[i] = fi
                
                # Check if we made any valid evaluations
                valid = np.isfinite(F)
                if not np.any(valid):
                    break
                
                # Update best
                min_idx = np.argmin(F)
                if F[min_idx] < best_f:
                    best_f = F[min_idx]
                    best_x = Y[min_idx].copy()
                
                # Sort by fitness
                order = np.argsort(F)
                Y_sorted = Y[order]
                F_sorted = F[order]
                
                # Update mean
                new_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
                
                # Update evolution paths
                y_vec = (new_mean - mean) / step_size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_vec
                norm_ps = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (it + 1)))
                
                if norm_ps < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)):
                    pc = (1 - cc) * pc + cc * y_vec
                
                # Update covariance
                delta_sigma = (norm_ps > (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
                C = ((1 - (mu + 2) * cmu) * C + 
                     np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu] - mean, Y_sorted[:mu] - mean), axis=0) +
                     (1 - (mu + 2) * cmu + 2 * cmu * mueff) * np.outer(pc, pc))
                
                # Ensure positive definiteness
                C = 0.5 * (C + C.T)
                C += 1e-12 * np.eye(dim)
                
                # Update mean and step size
                mean = new_mean
                step_size *= np.exp(cs / (dim + 1) * (norm_ps / np.sqrt(dim) - 1))
                step_size = np.clip(step_size, 1e-6, 10.0)
                
                # Update Cholesky factorization
                Csqrt = update_chol(C)
                
                # Convergence check
                if np.all(np.abs(Y_sorted[:mu] - mean) < 1e-7 * max(1, np.linalg.norm(mean))):
                    break
                
                # Check if step size is too small
                if step_size < 1e-7:
                    break
            
            # Between starts, do some local refinement around best point
            if best_x is not None and total_calls < budget and time_ok():
                # Simple local search around best point
                local_budget = min(20, budget - total_calls)
                for _ in range(local_budget):
                    if not time_ok():
                        break
                    perturbation = rng.standard_normal(dim) * 0.1
                    x_new = np.clip(best_x + perturbation, -5, 5)
                    f_new = safe_call(x_new)
                    if f_new is not None and f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
        
        # Use any remaining budget for final refinement
        if best_x is not None and total_calls < budget and time_ok():
            remaining = budget - total_calls
            for _ in range(min(remaining, 50)):
                if not time_ok():
                    break
                # Adaptive step size local search
                step = 0.05 * (1 + 0.5 * np.exp(-total_calls / 100))
                perturbation = rng.standard_normal(dim) * step
                x_new = np.clip(best_x + perturbation, -5, 5)
                f_new = safe_call(x_new)
                if f_new is not None and f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
