"""CMA-ES optimizer with LHS initialization for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with LHS initialization and deadline guard for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: LHS initialization to find a good starting point
        # Use a small number of points to quickly identify a promising region
        n_lhs = 10
        best_x = None
        best_f = np.inf

        # Generate LHS points
        for i in range(n_lhs):
            if time.time() > deadline:
                break

            x = np.zeros(dim)
            for d in range(dim):
                interval_start = i / n_lhs
                interval_end = (i + 1) / n_lhs
                x[d] = -5.0 + 10.0 * (interval_start + rng.random() * (interval_end - interval_start))
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES
        # Initialize CMA-ES parameters
        # Mean is the best point found in LHS
        mean = best_x.copy()
        # Initial sigma: heuristic based on domain size and dim
        sigma = 1.0 
        # Covariance matrix C
        C = np.eye(dim)
        # Step size adaptation state
        # We use a simplified CMA-ES implementation for speed and stability
        # Parameters for dim=5
        # Lambda (population size)
        lambda_ = max(4, 4 + int(3 * np.log(dim)))
        
        # Weights
        mu = lambda_ // 2
        weights = np.log(lambda_ + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial best
        current_best_x = best_x
        current_best_f = best_f
        remaining_budget = budget - n_lhs
        
        # Pre-compute Cholesky factor update logic
        # To avoid full Cholesky every step (expensive for small dim, but okay for 5),
        # we will update C directly using rank-1 and rank-mu updates.
        
        # We need to track the best solution found so far for the return value
        global_best_x = best_x
        global_best_f = best_f
        
        # CMA-ES Loop
        # We will consume the remaining budget.
        # Each iteration uses lambda_ function calls.
        # So max iterations = remaining_budget // lambda_
        
        # However, to maximize early progress and handle the deadline,
        # we might do a few more LHS-like explorations if budget allows, 
        # but CMA-ES is generally more efficient.
        
        # Let's run CMA-ES until budget or deadline is hit.
        
        while budget > 0 and time.time() < deadline:
            # Generate lambda_ samples
            # We need the Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes singular or non-positive definite, reset
                C = np.eye(dim)
                L = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 1.0
                
            samples = np.array([mean + sigma * (L @ rng.standard_normal(dim)) for _ in range(lambda_)])
            
            # Clip to bounds
            samples = np.clip(samples, -5, 5)
            
            # Evaluate samples
            fvals = np.array([func(s) for s in samples])
            
            # Update global best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < global_best_f:
                global_best_f = fvals[min_idx]
                global_best_x = samples[min_idx].copy()
            
            # Sort samples by fitness
            order = np.argsort(fvals)
            sorted_samples = samples[order]
            sorted_fvals = fvals[order]
            
            # Update mean (weighted average of top mu)
            new_mean = np.sum(weights[:, None] * sorted_samples[:mu], axis=0)
            
            # Update path p_sigma
            # Need C^-1/2
            try:
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
                
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (new_mean - mean) / sigma)
            
            # Update sigma (CSA - Cumulative Step-size Adaptation)
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * 1)) < (1.4 + 2 / (dim + 1)) * np.trace(np.eye(dim))) else 0
            # Simplified h_sigma check, standard CMA-ES uses a more complex check
            # For dim=5, this is usually stable.
            sigma = sigma * np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update path p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (new_mean - mean) / sigma)
            
            # Update covariance matrix C
            # Rank-1 update
            rank1 = np.outer(p_c, p_c)
            # Rank-mu update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_mu = delta_h * C
            for i in range(mu):
                y_i = C_inv_sqrt @ (sorted_samples[i] - mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * (2 - c_1) + c_mu - c_1) * C + c_1 * rank1 + c_mu * rank_mu
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2.0
            # Eigenvalue check for stability
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals <= 0):
                # Reset C if it becomes invalid
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 1.0
                break # Break out if unstable, fallback to simple search if needed, but usually reset is enough
            
            mean = new_mean
            
            # Update budget
            budget -= lambda_
            
            # Check if we've made enough progress or if sigma is too small
            if sigma < 1e-6:
                break

        # Final polish: if we have budget left and time, do a fine-grained local search
        # around the best point found by CMA-ES
        if budget > 0 and time.time() < deadline - 10:
            x = global_best_x.copy()
            fx = global_best_f
            sigma_local = max(1e-5, sigma * 0.1)
            while budget > 0 and time.time() < deadline - 5:
                y = x + sigma_local * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < fx:
                    x = y
                    fx = fy
                    sigma_local *= 1.05
                else:
                    sigma_local *= 0.95
                budget -= 1
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()
                if sigma_local < 1e-8:
                    break

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
