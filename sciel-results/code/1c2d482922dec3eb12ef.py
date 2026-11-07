"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with restarts for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        def time_left():
            return deadline - time.time()
            
        def cma_es(func, x0, f0, max_calls, deadline, dim, rng):
            """
            Run CMA-ES from x0 with f0.
            Returns (best_x, best_f, calls_used).
            """
            # Parameters
            # For dim=5, standard lambda is 4 + floor(3*log(5)) = 4+5=9. 
            # Let's use lambda=10 for a bit more exploration, mu=5.
            lam = 10
            mu = lam // 2
            
            # Initial step size
            sigma = 2.0 
            
            # CMA-ES state
            x_mean = x0.copy()
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            
            # Evolution path
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Damping for sigma update
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            # Hessian of log det C? No, we need trace(C) and trace(C_inv C) terms.
            # For simplicity in this implementation, we approximate the update.
            
            local_best_x = x0.copy()
            local_best_f = f0
            calls_used = 1
            
            # Initial simplex-like check? No, CMA-ES starts with a mean.
            # We need to evaluate the initial population.
            
            # History for stagnation detection
            best_f_history = [f0]
            stagnation_count = 0
            
            while calls_used < max_calls and time_left() > 0.5:
                # Generate population
                Y = rng.multivariate_normal(np.zeros(dim), C, size=lam)
                # More efficient: use Cholesky if we maintained it, but for dim=5, multivariate_normal is fine.
                # Actually, rng.multivariate_normal uses SVD/Cholesky internally.
                
                X = x_mean + sigma * Y
                # Clip to box? The problem says func clips, but CMA-ES works better in unbounded space.
                # However, if we go far out, it might be inefficient.
                # Let's clip X to [-5, 5] to keep evaluations "valid" in the context of the domain,
                # though func handles it. Clipping might distort the distribution.
                # Given the box is [-5, 5] and optimum is in [-4, 4], clipping is safe.
                X = np.clip(X, -5, 5)
                
                # Evaluate
                F = np.array([func(xi) for xi in X])
                calls_used += lam
                
                # Update global best
                idx_min = np.argmin(F)
                if F[idx_min] < local_best_f:
                    local_best_f = F[idx_min]
                    local_best_x = X[idx_min].copy()
                    
                # Check stagnation
                if len(best_f_history) > 10:
                    if local_best_f == best_f_history[-1]:
                        stagnation_count += 1
                    else:
                        stagnation_count = 0
                    best_f_history.append(local_best_f)
                    if len(best_f_history) > 100:
                        best_f_history.pop(0)
                    
                    if stagnation_count > 50:
                        break
                else:
                    best_f_history.append(local_best_f)
                
                # Sort by fitness
                order = np.argsort(F)
                X = X[order]
                F = F[order]
                
                # Update mean
                x_mean = np.sum(weights[:, None] * X[:mu], axis=0)
                
                # Update evolution path for sigma
                # p_sigma = (1-c_sigma)*p_sigma + c_sigma*sqrt(mu_eff)*C_inv_sqrt@(x_mean - old_mean)/sigma
                # We need C_inv_sqrt. For dim=5, we can compute it via eigen decomposition.
                eigvals, eigvecs = np.linalg.eigh(C)
                # Handle numerical issues
                eigvals = np.maximum(eigvals, 1e-10)
                C_inv_sqrt = eigvecs @ np.diag(1/np.sqrt(eigvals)) @ eigvecs.T
                
                y_mean = (x_mean - np.sum(weights[:, None] * X[:mu], axis=0) + sigma * Y[order[:mu]].mean(axis=0)) / sigma # Approximation, better to store old mean
                # Actually, standard update:
                # p_sigma = (1-c_sigma)*p_sigma + c_sigma*sqrt(mu_eff)*(x_mean - x_mean_old)/sigma
                # Let's stick to a simpler robust variant or standard one.
                # Standard CMA-ES:
                # x_mean_old = x_mean (before update)
                # p_sigma = (1-c_sigma)*p_sigma + c_sigma*sqrt(mu_eff)*(x_mean - x_mean_old)/sigma
                # This requires storing x_mean_old.
                
                # Let's refine the loop to store x_mean_old
                # (Refactoring inside the loop is messy, let's just use a simpler covariance update for this specific dim)
                
                # Simplified CMA-ES update for low dim:
                # 1. Update mean
                # 2. Update covariance: C = (1-c_1-c_mu)*C + c_1*(p_c p_c^T + (1-delta_sigma)*C) + c_mu * sum(w_i y_i y_i^T)
                # where y_i = (x_i - x_mean_old)/sigma
                
                # To keep code clean and robust, I will use a slightly simplified update rule
                # that is known to work well for small dims.
                
                # Recalculate C_inv_sqrt for the path update if needed, or just skip path update for simplicity?
                # No, path is important.
                
                # Let's do the standard update properly.
                # We need x_mean_old.
                # I will restructure the loop slightly in the final code.
                
            return local_best_x, local_best_f, calls_used

        # Since the internal CMA-ES implementation above is getting complex and prone to subtle bugs
        # in the path updates without careful state management, I will implement a cleaner version
        # specifically tailored for dim=5.
        
        def cma_es_clean(func, x0, f0, max_calls, deadline, dim, rng):
            lam = 10
            mu = 5
            
            # Initial step size
            sigma = 2.0 
            
            x_mean = x0.copy()
            C = np.eye(dim)
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            local_best_x = x0.copy()
            local_best_f = f0
            calls_used = 1
            
            best_f_history = []
            stagnation_count = 0
            
            while calls_used + lam < max_calls and time_left() > 0.5:
                x_mean_old = x_mean.copy()
                
                # Sample
                # Use Cholesky decomposition of C for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, regularize
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-6)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    L = np.linalg.cholesky(C)
                    
                Y = rng.normal(size=(lam, dim)) @ L.T
                X = x_mean + sigma * Y
                X = np.clip(X, -5, 5)
                
                # Evaluate
                F = np.array([func(xi) for xi in X])
                calls_used += lam
                
                # Update global best
                idx_min = np.argmin(F)
                if F[idx_min] < local_best_f:
                    local_best_f = F[idx_min]
                    local_best_x = X[idx_min].copy()
                    
                # Stagnation check
                if len(best_f_history) > 0:
                    if local_best_f <= best_f_history[-1] + 1e-10:
                        stagnation_count += 1
                    else:
                        stagnation_count = 0
                best_f_history.append(local_best_f)
                if len(best_f_history) > 50:
                    best_f_history.pop(0)
                    
                if stagnation_count > 30:
                    break
                
                # Sort
                order = np.argsort(F)
                X = X[order]
                F = F[order]
                
                # Update mean
                x_mean = np.sum(weights[:, None] * X[:mu], axis=0)
                
                # Update evolution paths
                # p_sigma
                # Need C_inv_sqrt
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C_inv_sqrt = eigvecs @ np.diag(1/np.sqrt(eigvals)) @ eigvecs.T
                
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * C_inv_sqrt @ ((x_mean - x_mean_old) / sigma)
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # p_c
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used - 1) / lam)) < (1.4 + 2 / (dim + 1)) * chi_n
                p_c = (1 - c_c) * p_c + h_sigma * c_c * ((x_mean - x_mean_old) / sigma)
                
                # Update covariance
                delta_sigma = (1 - (1 - c_sigma)**2) * (np.linalg.norm(p_sigma)**2 / chi_n**2 - 1)
                rank_one = (1 - c_1) * C + c_1 * (p_c @ p_c.T + (1 - delta_sigma) * C)
                
                # Rank-mu update
                Y_mu = (X[:mu] - x_mean_old) / sigma
                rank_mu = c_mu * np.sum(weights[:, None, None] * (Y_mu[:, :, None] * Y_mu[:, None, :]), axis=0)
                
                C = rank_one + rank_mu
                
                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 1e-6):
                    eigvals = np.maximum(eigvals, 1e-6)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    
            return local_best_x, local_best_f, calls_used

        # Phase 1: Global Search
        n_random = 20
        candidates = []
        for i in range(n_random):
            if calls_made >= budget or time_left() < 1.0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Sort candidates
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-start CMA-ES
        # Select diverse starts
        selected_starts = []
        used_points = [best_x]
        for fx, x in candidates:
            if len(selected_starts) >= 3:
                break
            is_diverse = True
            for existing in used_points:
                if np.linalg.norm(x - existing) < 1.5:
                    is_diverse = False
                    break
            if is_diverse:
                selected_starts.append((fx, x))
                used_points.append(x)
                
        if not selected_starts:
            selected_starts = [(best_f, best_x)]
            
        n_starts = len(selected_starts)
        
        for i, (fx_start, x_start) in enumerate(selected_starts):
            if calls_made >= budget or time_left() < 5.0:
                break
            
            # Allocate budget
            # Give each start a chunk. If one fails (stagnates), the rest of the budget is wasted.
            # To mitigate this, we can run them in parallel? No, sequential is safer for budget tracking.
            # Let's allocate 1/3 of remaining budget to each, but cap it.
            remaining = budget - calls_made
            alloc = remaining // (n_starts - i)
            if alloc < 20:
                alloc = remaining
                
            # Run CMA-ES
            x_best, f_best, used = cma_es_clean(func, x_start, fx_start, alloc, deadline, dim, rng)
            calls_made += used
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
                
        return best_x, best_f

# EVOLVE-BLOCK-END
