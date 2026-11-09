"""Rewritten program using Population-Based CMA-ES for rotated Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for rotated Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280  # Safety margin

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            try:
                val = func(np.asarray(x, dtype=float).reshape(dim))
                if np.isfinite(val):
                    return float(val)
                return 1e10
            except Exception:
                return 1e110

        def cma_es(start_x, max_calls):
            """
            Compact CMA-ES implementation.
            Returns the best (x, f) found within max_calls.
            """
            nonlocal best_x, best_f, call_count
            
            # CMA-ES Parameters
            # lambda is population size
            lam = 4 
            mu = lam // 2  # Weighting: use best half
            # Weights
            weights = np.zeros(lam)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Adaptation rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            # Initial values
            x = np.clip(start_x, -5, 5).copy()
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            sigma = 2.0  # Large initial sigma to explore
            C = np.eye(dim)
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            
            # State vectors
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Cache for square root (simplified, we will use cholesky or eigen if needed, 
            # but for dim=5, we can just use np.linalg.inv and cholesky every step 
            # or use a simpler update. For robustness and speed in Python, 
            # we will update C and C_inv, and sample using a fixed basis 
            # scaled by sigma, then transform. 
            # Actually, a simple (mu+1)-CMA or just using the population 
            # mean and covariance is standard.
            # To keep it fast in Python, we will update C every 10 steps 
            # or use a simplified gradient-free approach? 
            # No, let's stick to standard CMA-ES but optimize the loop.
            # For dim=5, matrix operations are cheap.
            
            n_evals = 1
            x_mean = x.copy()
            fitnesses = np.array([fx])
            xs = [x.copy()]
            
            # We need to collect 'lam' points per generation.
            # Since budget is tight, we might not finish full generations.
            # We will run until max_calls or budget.
            
            while call_count < budget and time.time() < deadline and (call_count - n_evals + lam) <= max_calls + lam:
                # Generate population
                for i in range(lam):
                    if call_count >= budget or time.time() > deadline:
                        break
                        
                    # Sample: x_mean + sigma * C_sqrt * z
                    z = rng.standard_normal(dim)
                    # Approximate C_sqrt using the current C. 
                    # For dim=5, we can compute cholesky or just use 
                    # a simple transform. We'll maintain C_sqrt via 
                    # eigen-decomposition or just use np.linalg.cholesky.
                    # To avoid expensive sqrt every step, we can update 
                    # C_sqrt less frequently, but for correctness, let's 
                    # just do it. Dim 5 is tiny.
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)
                    
                    y = L @ z
                    xi = np.clip(x_mean + sigma * y, -5, 5)
                    fi = safe_func(xi)
                    
                    xs.append(xi)
                    fitnesses = np.append(fitnesses, fi)
                    n_evals += 1
                    
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                
                if len(xs) < lam:
                    break
                    
                # Sort by fitness
                indices = np.argsort(fitnesses)
                xs_sorted = [xs[i] for i in indices]
                fs_sorted = fitnesses[indices]
                
                # Update mean
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * xs_sorted[i]
                    
                # Update sigma adaptation (CSA)
                # h_sigma condition
                h_sigma = 1.0
                # Calculate expected norm of p_sigma update
                # This is complex, so we simplify: just use the standard update
                # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * (1/ sigma) * C^{-1/2} (x_w - x_mean)
                # We need C^{-1/2}. We have L such that C = L L^T. So C^{-1/2} is L^{-T}.
                # We can solve L^T v = (x_w - x_mean) / sigma.
                
                x_w = x_new
                diff = (x_w - x_mean) / sigma
                # Solve L^T v = diff -> v = L^{-T} diff
                try:
                    v = np.linalg.solve(L.T, diff)
                except:
                    v = diff
                    
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v
                p_sigma_norm = np.linalg.norm(p_sigma)
                # h_sigma heuristic: if p_sigma_norm is not too large
                # chi_n = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                if p_sigma_norm / chi_n < 1.4 + 2/(dim+1):
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0
                    
                sigma = sigma * np.exp(c_sigma / chi_n * (p_sigma_norm / chi_n - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Update path pc and covariance C
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_w - x_mean) / sigma
                
                # Delta h term
                delta_h = (1 - h_sigma) * (1 - (1 - c_c) ** (2 * n_evals))
                
                # Outer product for C update
                # C = (1 + c_1 + c_mu - c_c * (1 - c_c)) C + c_1 (pc pc^T - delta_h C) + c_mu sum w_i (y_i y_i^T)
                # where y_i = (x_i - x_mean) / sigma
                
                # Calculate y_i for top mu
                y_mats = []
                for i in range(mu):
                    y_i = (xs_sorted[i] - x_mean) / sigma
                    y_mats.append(np.outer(y_i, y_i))
                    
                # Update C
                C_new = (1 + c_1 + c_mu - c_c * (1 - c_c)) * C
                C_new += c_1 * (np.outer(p_c, p_c) - delta_h * C)
                for i in range(mu):
                    C_new += c_mu * weights[i] * y_mats[i]
                    
                # Ensure symmetry and positive definiteness
                C = (C_new + C_new.T) / 2.0
                
                # Update C_mean (recentering to keep mean at 0? No, we just update mean)
                x_mean = x_new
                
                # Reset for next generation
                xs = []
                fitnesses = np.array([])
                
                # Stagnation check: if no improvement in best_f for a while, 
                # we might want to break to save budget for other restarts.
                # We'll let the outer loop handle restarts.
                
                if n_evals >= max_calls:
                    break

            return best_x, best_f

        # Multi-restart strategy
        # We divide the budget among a few restarts to handle multi-modality.
        # But CMA-ES is expensive per generation. 
        # In 5D, a generation takes 4 evals + 1 mean = 5 evals.
        # 500 evals / 5 = 100 generations.
        # A single CMA-ES run might be sufficient if it's well-tuned.
        # However, Rastrigin is very rugged.
        # Let's try 3 restarts with 150 evals each, plus a final polish.
        
        n_restarts = 3
        budget_per_restart = budget // (n_restarts + 1)
        
        # Start points
        starts = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-3, 3, dim)
        ]
        
        for i in range(n_restarts):
            if time.time() > deadline or call_count >= budget:
                break
            cma_es(starts[i], budget_per_restart)
            
        # Final polish with a smaller sigma CMA-ES or simple 1+1 around best
        if best_x is not None and call_count < budget and time.time() < deadline:
             # Simple 1+1 ES polish with small sigma
             x = best_x.copy()
             fx = best_f
             sigma = 0.2
             while call_count < budget and time.time() < deadline:
                 y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                 fy = safe_func(y)
                 if fy < fx:
                     x, fx = y, fy
                     sigma *= 1.1
                     if fx < best_f:
                         best_f = fx
                         best_x = x.copy()
                 else:
                     sigma *= 0.9
                     if sigma < 0.01:
                         break
                         
        return best_x, best_f
# EVOLVE-BLOCK-END
